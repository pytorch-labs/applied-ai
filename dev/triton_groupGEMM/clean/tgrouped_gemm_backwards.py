# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Tuple

import torch
import triton
import triton.language as tl
from tma_utils import TmaAutoTuneHelper

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@triton.jit
def _kernel_grouped_gemm_backward_x_scheduled(
    grad_y_ptr,  # grad of dl/dY [M, N*G]
    w_t_ptr,  # w transposed [K, N*G]
    grad_x_ptr,  # output of kernel [M, K]
    group_offsets_ptr,  # Pre-computed group offsets [G+1]
    workspace,  # Workspace for TMA descriptors
    G,  # Number of groups
    M,  # Total M dimension size
    N,  # N per group
    K,  # K dimension size
    stride_go_m,  # Stride for grad_output in M dimension
    stride_go_n,  # Stride for grad_output in N dimension
    stride_w_n,  # Stride for weights in N dimension
    stride_w_k,  # Stride for weights in K dimension
    stride_gx_m,  # Stride for grad_x in M dimension
    stride_gx_k,  # Stride for grad_x in K dimension
    NUM_SMS,  # Number of SMs on the GPU
    USE_TMA_LOAD: tl.constexpr = False,
    USE_TMA_STORE: tl.constexpr = False,
    BLOCK_SIZE_M: tl.constexpr = 64,
    BLOCK_SIZE_N: tl.constexpr = 64,
    BLOCK_SIZE_K: tl.constexpr = 64,
    GROUP_SIZE_M: tl.constexpr = 8,
    EVEN_K: tl.constexpr = True,
) -> None:
    """
    Scheduled grouped GEMM backward for X with TMA support.

    For each group g, computes: grad_x[g] = grad_y[g] @ w_t[g].T

    Where:
    - grad_y is [M, N*G]
    - w_t is [K, N*G] (transposed from [N*G, K])
    - grad_x is [M, K]
    """
    # Get coordinates for the current program
    tidx = tl.program_id(axis=0)
    dtype = grad_x_ptr.dtype.element_ty
    TMA_SIZE: tl.constexpr = 128

    # Initialize workspace pointer if using TMA store
    if USE_TMA_STORE:
        c_desc_ptr = workspace + tidx * TMA_SIZE
    else:
        c_desc_ptr = None

    # Calculate work distribution
    pid = tidx
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = num_pid_m * num_pid_k
    group_id = pid // num_pid_in_group
    pid_in_group = pid % num_pid_in_group
    pid_m = pid_in_group % num_pid_m
    pid_k = pid_in_group // num_pid_m

    # Get group boundaries
    valid_group = group_id < G
    group_start = tl.where(valid_group, tl.load(group_offsets_ptr + group_id), 0)
    group_end = tl.where(valid_group, tl.load(group_offsets_ptr + group_id + 1), 0)
    group_size = group_end - group_start

    # Calculate a mask for valid processing (valid group and non-empty)
    valid_work = valid_group & (group_size > 0)

    # Only process if we have valid work
    if valid_work:
        # Compute offsets for this group
        n_start = group_id * N

        # Block dimensions
        m_block_offset = pid_m * BLOCK_SIZE_M
        k_block_offset = pid_k * BLOCK_SIZE_K

        # Setup TMA descriptor for output if using TMA
        if USE_TMA_STORE:
            m_size = tl.minimum(
                BLOCK_SIZE_M, group_end - (group_start + m_block_offset)
            )
            if m_size > 0:
                tl.extra.cuda.experimental_device_tensormap_create2d(
                    desc_ptr=c_desc_ptr,
                    global_address=grad_x_ptr
                    + (group_start + m_block_offset) * K
                    + k_block_offset,
                    load_size=[m_size, tl.minimum(BLOCK_SIZE_K, K - k_block_offset)],
                    global_size=[m_size, K],
                    element_ty=dtype,
                )
                tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(c_desc_ptr)

        # Initialize offsets for this block
        offs_m = group_start + m_block_offset + tl.arange(0, BLOCK_SIZE_M)

        # For K dimension, optimize memory access if EVEN_K is True
        offs_k = k_block_offset + tl.arange(0, BLOCK_SIZE_K)

        # Create masks
        m_mask = offs_m < group_end
        k_mask = offs_k < K

        # Initialize accumulator
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

        # Loop over the reduction dimension (N)
        for k_offset in range(0, N, BLOCK_SIZE_N):
            # Handle boundary conditions for the reduction dimension
            k_size = tl.minimum(BLOCK_SIZE_N, N - k_offset)
            offs_n = n_start + k_offset + tl.arange(0, BLOCK_SIZE_N)
            n_mask = offs_n < (n_start + N)

            # Load grad_y [M, N*G] block
            grad_y_block = tl.load(
                grad_y_ptr + offs_m[:, None] * (N * G) + offs_n[None, :],
                mask=m_mask[:, None] & n_mask[None, :],
                other=0.0,
            )

            # Load w_t [K, N*G] block
            w_t_block = tl.load(
                w_t_ptr + offs_k[:, None] * (N * G) + offs_n[None, :],
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )

            # grad_y @ w_t.T
            # Allow TF32 for higher performance if K is even and divisible by 8
            # this may not be required for the backward pass
            if EVEN_K:
                accumulator += tl.dot(
                    grad_y_block.to(tl.float32),
                    w_t_block.to(tl.float32).T,
                    allow_tf32=True,
                )
            else:
                accumulator += tl.dot(
                    grad_y_block.to(tl.float32),
                    w_t_block.to(tl.float32).T,
                    allow_tf32=False,
                )

        # Store result to grad_x
        if USE_TMA_STORE:
            m_offset = 0
            n_offset = 0

            # Convert to output dtype and store
            tl._experimental_descriptor_store(
                c_desc_ptr,
                accumulator.to(dtype),
                [m_offset, n_offset],
            )
        else:
            # Standard store
            tl.store(
                grad_x_ptr + offs_m[:, None] * K + offs_k[None, :],
                accumulator.to(dtype),
                mask=m_mask[:, None] & k_mask[None, :],
            )

        tidx += NUM_SMS


@triton.jit
def _kernel_grouped_gemm_backward_w_scheduled(
    x_t_ptr,  # x transposed [K, M]
    grad_y_ptr,  # grad of dl/dY [M, N*G]
    grad_w_ptr,  # output of kernel (grad_w) [N*G, K]
    group_offsets_ptr,  # Pre-computed group offsets [G+1]
    workspace,  # Workspace for TMA descriptors
    G,  # Number of groups
    M,  # Total M dimension size
    N,  # N per group
    K,  # K dimension size
    stride_x_m,  # Stride for x in M dimension
    stride_x_k,  # Stride for x in K dimension
    stride_go_m,  # Stride for grad_output in M dimension
    stride_go_n,  # Stride for grad_output in N dimension
    stride_gw_n,  # Stride for grad_w in N dimension
    stride_gw_k,  # Stride for grad_w in K dimension
    NUM_SMS,  # Number of SMs on the GPU
    USE_TMA_LOAD: tl.constexpr = False,
    USE_TMA_STORE: tl.constexpr = False,
    BLOCK_SIZE_N: tl.constexpr = 64,
    BLOCK_SIZE_K: tl.constexpr = 64,
    BLOCK_SIZE_M: tl.constexpr = 32,
    GROUP_SIZE_N: tl.constexpr = 8,
    EVEN_K: tl.constexpr = True,
) -> None:
    """
    Scheduled implementation of grouped GEMM backward for W with TMA support.

    For each group g, computes:
        grad_w[g] = grad_y[g].T @ x[g]

    Where:
    - x_t is [K, M] (transposed from [M, K])
    - grad_y is [M, N*G]
    - grad_w is [N*G, K]
    """
    # Define coordinates for the current program
    tidx = tl.program_id(axis=0)
    dtype = grad_w_ptr.dtype.element_ty
    TMA_SIZE: tl.constexpr = 128

    # Initialize workspace pointer if using TMA store
    if USE_TMA_STORE:
        c_desc_ptr = workspace + tidx * TMA_SIZE
    else:
        c_desc_ptr = None

    # Calculate work distribution
    pid = tidx
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = num_pid_n * num_pid_k
    group_id = pid // num_pid_in_group
    pid_in_group = pid % num_pid_in_group
    pid_n = pid_in_group % num_pid_n
    pid_k = pid_in_group // num_pid_n

    # Get group boundaries
    valid_group = group_id < G
    group_start = tl.where(valid_group, tl.load(group_offsets_ptr + group_id), 0)
    group_end = tl.where(valid_group, tl.load(group_offsets_ptr + group_id + 1), 0)
    group_size = group_end - group_start

    # Calculate a mask for valid processing (valid group and non-empty)
    valid_work = valid_group & (group_size > 0)

    # Only process if we have valid work
    if valid_work:
        # Compute offsets for this group
        n_start = group_id * N

        # Block dimensions
        n_block_offset = pid_n * BLOCK_SIZE_N
        k_block_offset = pid_k * BLOCK_SIZE_K

        # Setup TMA descriptor for output if using TMA
        if USE_TMA_STORE:
            n_size = tl.minimum(BLOCK_SIZE_N, N - n_block_offset)
            if n_size > 0:
                tl.extra.cuda.experimental_device_tensormap_create2d(
                    desc_ptr=c_desc_ptr,
                    global_address=grad_w_ptr
                    + (n_start + n_block_offset) * K
                    + k_block_offset,
                    load_size=[n_size, tl.minimum(BLOCK_SIZE_K, K - k_block_offset)],
                    global_size=[n_size, K],
                    element_ty=dtype,
                )
                tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(c_desc_ptr)

        # Initialize offsets for this block
        offs_n = n_start + n_block_offset + tl.arange(0, BLOCK_SIZE_N)
        offs_k = k_block_offset + tl.arange(0, BLOCK_SIZE_K)

        # Create masks
        n_mask = offs_n < (n_start + N)
        k_mask = offs_k < K

        # Initialize accumulator
        accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)

        # Loop over the reduction dimension (M)
        for m_offset in range(0, group_size, BLOCK_SIZE_M):
            # Handle boundary conditions for the reduction dimension
            m_size = tl.minimum(BLOCK_SIZE_M, group_size - m_offset)
            offs_m = group_start + m_offset + tl.arange(0, BLOCK_SIZE_M)
            m_mask = offs_m < group_end

            # Load grad_y [M, N*G] block
            grad_y_block = tl.load(
                grad_y_ptr + offs_m[:, None] * (N * G) + offs_n[None, :],
                mask=m_mask[:, None] & n_mask[None, :],
                other=0.0,
            )

            # Load x_t [K, M] block
            x_t_block = tl.load(
                x_t_ptr + offs_k[:, None] * M + offs_m[None, :],
                mask=k_mask[:, None] & m_mask[None, :],
                other=0.0,
            )

            # Matrix multiplication: (grad_y_block.T @ x_t_block.T)
            # Allow TF32 for higher performance if K is even and divisible by 8
            # this may not be required for the backward pass
            if EVEN_K:
                accumulator += tl.dot(
                    grad_y_block.to(
                        tl.float32
                    ).T,  # Shape: [BLOCK_SIZE_N, BLOCK_SIZE_M]
                    x_t_block.to(tl.float32).T,  # Shape: [BLOCK_SIZE_M, BLOCK_SIZE_K]
                    allow_tf32=True,
                )
            else:
                accumulator += tl.dot(
                    grad_y_block.to(
                        tl.float32
                    ).T,  # Shape: [BLOCK_SIZE_N, BLOCK_SIZE_M]
                    x_t_block.to(tl.float32).T,  # Shape: [BLOCK_SIZE_M, BLOCK_SIZE_K]
                    allow_tf32=False,
                )

        # Store result to grad_w
        if USE_TMA_STORE:

            n_offset = 0
            k_offset = 0

            # Convert to output dtype and store
            tl._experimental_descriptor_store(
                c_desc_ptr,
                accumulator.to(dtype),
                [n_offset, k_offset],
            )
        else:
            # Standard store
            tl.store(
                grad_w_ptr + offs_n[:, None] * K + offs_k[None, :],
                accumulator.to(dtype),
                mask=n_mask[:, None] & k_mask[None, :],
            )

        tidx += NUM_SMS


def _compute_grad_x_pytorch(grad_output, w, m_sizes, grad_x):
    """
    Compute grad_x using pure PyTorch operations with FP32 precision for better accuracy.
    """
    G = m_sizes.shape[0]
    M, K = grad_x.shape
    N = w.shape[0] // G

    # Zero out the output tensor first
    grad_x.zero_()

    # Store original dtype and convert to float32 for computation
    orig_dtype = grad_x.dtype
    grad_output_fp32 = grad_output.float()
    w_fp32 = w.float()
    grad_x_fp32 = torch.zeros_like(grad_x, dtype=torch.float32)

    # Process each group separately
    m_start = 0
    for g in range(G):
        m_size = m_sizes[g].item()
        if m_size > 0:
            m_end = m_start + m_size
            n_start = g * N
            n_end = (g + 1) * N

            # Get slices for this group
            grad_output_slice = grad_output_fp32[m_start:m_end, n_start:n_end]
            w_slice = w_fp32[n_start:n_end]

            # Process in chunks for better precision on large matrices
            CHUNK_SIZE = 256
            for chunk_start in range(0, m_size, CHUNK_SIZE):
                chunk_end = min(chunk_start + CHUNK_SIZE, m_size)
                chunk_size = chunk_end - chunk_start

                # Compute matrix multiplication with higher precision
                grad_output_chunk = grad_output_slice[chunk_start:chunk_end]
                result_chunk = torch.matmul(
                    grad_output_chunk.double(), w_slice.double()
                )

                # Store the result
                grad_x_fp32[m_start + chunk_start : m_start + chunk_end].copy_(
                    result_chunk.float()
                )

        m_start = m_end

    # Convert back to original dtype
    grad_x.copy_(grad_x_fp32.to(orig_dtype))


def _compute_grad_w_pytorch(grad_output, x, m_sizes, grad_w):
    """
    Compute grad_w using pure PyTorch operations with FP64 precision for better accuracy.
    """
    G = m_sizes.shape[0]
    N_times_G, K = grad_w.shape
    N = N_times_G // G

    # Zero out the output tensor first
    grad_w.zero_()

    # Store original dtype and convert to float32 for computation
    orig_dtype = grad_w.dtype
    grad_output_fp32 = grad_output.float()
    x_fp32 = x.float()
    grad_w_fp32 = torch.zeros_like(grad_w, dtype=torch.float32)

    # Handle potential K dimension mismatches
    K_x = x.shape[1]
    min_K = min(K, K_x)

    # Process each group separately
    m_start = 0
    for g in range(G):
        m_size = m_sizes[g].item()
        if m_size > 0:
            m_end = m_start + m_size
            n_start = g * N
            n_end = (g + 1) * N

            # Get slices for this group
            grad_output_slice = grad_output_fp32[m_start:m_end, n_start:n_end]
            x_slice = x_fp32[m_start:m_end, :min_K]

            # Process in chunks for better precision
            CHUNK_SIZE = 32
            result = torch.zeros(
                (grad_output_slice.shape[1], min_K),
                dtype=torch.float64,
                device=grad_output_slice.device,
            )

            for chunk_start in range(0, m_size, CHUNK_SIZE):
                chunk_end = min(chunk_start + CHUNK_SIZE, m_size)

                # Get chunks
                grad_output_chunk = grad_output_slice[chunk_start:chunk_end].double()
                x_chunk = x_slice[chunk_start:chunk_end].double()

                # Matrix multiplication in FP64
                chunk_result = torch.matmul(grad_output_chunk.t(), x_chunk)
                result += chunk_result

            # Handle K dimension padding if needed
            if K > min_K:
                temp_result = torch.zeros(
                    (grad_output_slice.shape[1], K),
                    dtype=torch.float32,
                    device=grad_output_slice.device,
                )
                temp_result[:, :min_K] = result.float()
                grad_w_fp32[n_start:n_end].copy_(temp_result)
            else:
                grad_w_fp32[n_start:n_end].copy_(result.float())

        m_start = m_end

    # Convert back to original dtype
    grad_w.copy_(grad_w_fp32.to(orig_dtype))


def _pytorch_fallback_backward(grad_output, x, w, m_sizes):
    """
    Pure PyTorch implementation of grouped GEMM backward with high precision.
    Used as a fallback when the Triton kernels cannot be used.
    """
    logging.info(
        "WARNING:  Using PyTorch fallback for grouped GEMM backward with high precision"
    )

    # Ensure inputs are contiguous
    x = x.contiguous()
    w = w.contiguous()
    grad_output = grad_output.contiguous()
    m_sizes = m_sizes.contiguous()

    # Allocate output tensors
    grad_x = torch.zeros_like(x)
    grad_w = torch.zeros_like(w)

    # Compute gradients using the helper functions
    _compute_grad_x_pytorch(grad_output, w, m_sizes, grad_x)
    _compute_grad_w_pytorch(grad_output, x, m_sizes, grad_w)

    return grad_x, grad_w


def _pytorch_reference_backward(grad_output, x, w, m_sizes):
    """
    Pure PyTorch implementation of grouped GEMM backward for validation.
    Simple version that's easy to verify but may be less numerically accurate
    for large matrices.
    """
    # Create output gradients
    grad_x = torch.zeros_like(x)
    grad_w = torch.zeros_like(w)

    # Compute group-by-group
    G = m_sizes.shape[0]
    N = w.shape[0] // G

    m_start = 0
    for g in range(G):
        m_size = m_sizes[g].item()
        if m_size > 0:
            m_end = m_start + m_size
            n_start = g * N
            n_end = (g + 1) * N

            # Compute gradients
            grad_x[m_start:m_end] = torch.matmul(
                grad_output[m_start:m_end, n_start:n_end], w[n_start:n_end]
            )
            grad_w[n_start:n_end] = torch.matmul(
                grad_output[m_start:m_end, n_start:n_end].t(), x[m_start:m_end]
            )

        m_start += m_size

    return grad_x, grad_w


def grouped_gemm_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    m_sizes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Backward pass for grouped matrix multiplication using scheduled kernels with TMA support.

    Args:
        grad_output: Gradient with respect to output, shape [M, N*G]
        x: Input tensor from forward pass, shape [M, K]
        w: Weight tensor from forward pass, shape [N*G, K]
        m_sizes: Group sizes tensor, shape [G]

    Returns:
        Tuple of gradients with respect to x and w: (grad_x, grad_w)
    """
    logging.info("Starting grouped_gemm_backward with TMA-enabled scheduling")

    # Check CUDA availability
    if not torch.cuda.is_available():
        logging.error("CUDA not available for backward pass")
        return _pytorch_fallback_backward(grad_output, x, w, m_sizes)

    # Get GPU parameters
    device_props = torch.cuda.get_device_properties("cuda")
    NUM_SMS = device_props.multi_processor_count

    # Check TMA support
    has_tma = hasattr(tl.extra, "cuda") and device_props.major >= 9

    if has_tma:
        logging.info(f"TMA support detected on GPU with {NUM_SMS} SMs")
        USE_TMA_LOAD = True
        USE_TMA_STORE = False  # Default to disabled until verified reliability
    else:
        logging.warning("TMA support not detected, disabling TMA optimizations")
        USE_TMA_LOAD = False
        USE_TMA_STORE = False

    # Validate input dimensions
    G = m_sizes.shape[0]
    M, K_x = x.shape
    N_times_G, K_w = w.shape

    # Check that K dimensions match
    if K_x != K_w:
        logging.warning(f"K dimension mismatch: x has K={K_x}, w has K={K_w}")
        return _pytorch_fallback_backward(grad_output, x, w, m_sizes)

    try:
        # Ensure contiguous tensors
        grad_output = grad_output.contiguous()
        x = x.contiguous()
        w = w.contiguous()
        m_sizes = m_sizes.contiguous()

        # Allocate output tensors
        grad_x = torch.zeros_like(x)
        grad_w = torch.zeros_like(w)

        # Determine N per group
        # Note that DeepSeek uses M*G instead of N*G
        N = N_times_G // G

        # Set strided access patterns
        stride_go_m = N * G  # Stride for grad_output in M dimension
        stride_go_n = 1  # Stride for grad_output in N dimension

        stride_w_n = K_x  # Stride for weights in N dimension
        stride_w_k = 1  # Stride for weights in K dimension

        stride_gx_m = K_x  # Stride for grad_x in M dimension
        stride_gx_k = 1  # Stride for grad_x in K dimension

        stride_x_m = K_x  # Stride for x in M dimension
        stride_x_k = 1  # Stride for x in K dimension

        stride_gw_n = K_w  # Stride for grad_w in N dimension
        stride_gw_k = 1  # Stride for grad_w in K dimension

        # Pre-compute group offsets for efficient group indexing
        group_offsets = torch.zeros(G + 1, device=m_sizes.device, dtype=torch.int32)
        m_offset = 0
        for g in range(G):
            group_offsets[g] = m_offset
            m_offset += m_sizes[g].item()
        group_offsets[G] = m_offset  # Total M

        # Check if K dimension is even (can optimize memory access patterns?)
        EVEN_K = (K_x % 8) == 0
        logging.info(f"EVEN_K optimization enabled: {EVEN_K} (K={K_x})")

        # Transpose x and w for backward computation
        x_t = x.T.contiguous()  # Shape: [K, M]
        w_t = w.T.contiguous()  # Shape: [K, N*G]

        # Allocate workspace for TMA descriptors
        workspace = torch.empty((NUM_SMS * 128), device=x.device, dtype=torch.uint8)

        try:
            logging.info("Computing grad_x with TMA-enabled kernel")

            # Fixed grid size based on SM count
            grid = (NUM_SMS,)

            _kernel_grouped_gemm_backward_x_scheduled[grid](
                grad_output,
                w_t,  # Using transposed weights
                grad_x,
                group_offsets,
                workspace,
                G,
                M,
                N,
                K_x,
                stride_go_m,
                stride_go_n,
                stride_w_n,
                stride_w_k,
                stride_gx_m,
                stride_gx_k,
                NUM_SMS,
                USE_TMA_LOAD,
                USE_TMA_STORE,
                EVEN_K=EVEN_K,
            )
            logging.info(
                "SUCCESS!! grad_X computation successful with TMA-enabled kernel"
            )
        except Exception as e:
            logging.error(f"FAILED: Error in TMA-enabled backward_x kernel: {e}")
            logging.info("WARNING: Falling back to PyTorch for grad_x")
            _compute_grad_x_pytorch(grad_output, w, m_sizes, grad_x)

        try:
            logging.info("Computing grad_w with TMA-enabled kernel")

            # Fixed grid size based on SM count
            grid = (NUM_SMS,)

            _kernel_grouped_gemm_backward_w_scheduled[grid](
                x_t,  # Using transposed inputs
                grad_output,
                grad_w,
                group_offsets,
                workspace,
                G,
                M,
                N,
                K_w,
                stride_x_m,
                stride_x_k,
                stride_go_m,
                stride_go_n,
                stride_gw_n,
                stride_gw_k,
                NUM_SMS,
                USE_TMA_LOAD,
                USE_TMA_STORE,
                EVEN_K=EVEN_K,
            )
            logging.info(
                "SUCCESS!! grad_W computation successful with TMA-enabled kernel"
            )
        except Exception as e:
            logging.error(f"FAILED:  Error in TMA-enabled backward_w kernel: {e}")
            logging.info("WARNING: Falling back to PyTorch for grad_w")
            _compute_grad_w_pytorch(grad_output, x, m_sizes, grad_w)

        return grad_x, grad_w
    except Exception as e:
        logging.error(f"Error in grouped_gemm_backward: {e}")
        return _pytorch_fallback_backward(grad_output, x, w, m_sizes)
