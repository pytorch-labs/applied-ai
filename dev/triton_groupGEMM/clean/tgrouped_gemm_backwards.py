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


# Fixed configuration for H100 - no autotuning
# These are conservative settings that should work well on H100 GPUs
@triton.jit
def _kernel_grouped_gemm_backward_w(
    x_t_ptr,  # x transposed [K, M]
    grad_y_ptr,  # grad of dl/dY [M, N*G]
    grad_w_ptr,  # output of kernel (grad_w) [N*G, K]
    workspace,
    m_sizes,
    # problem sizes
    G: tl.constexpr,
    M_bucket: tl.constexpr,
    N: tl.constexpr,  # N is per group
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    USE_TMA_LOAD: tl.constexpr,
    USE_TMA_STORE: tl.constexpr,
    # tile sizes - fixed for H100
    BLOCK_SIZE_M: tl.constexpr = 64,
    BLOCK_SIZE_N: tl.constexpr = 64,
    BLOCK_SIZE_K: tl.constexpr = 32,
) -> None:
    """
    Compute gradients with respect to w (weights).

    grad_w = x_t (x transposed) @ grad_y (dl/dY)

    Here:
    - x_t is [K, M] (transposed from [M, K])
    - grad_y is [M, N*G] where N is output dim per group
    - grad_w is [N*G, K]
    """
    tidx = tl.program_id(0)
    dtype = grad_w_ptr.dtype.element_ty
    TMA_SIZE: tl.constexpr = 128

    # Initialize workspace pointer if using TMA store
    if USE_TMA_STORE:
        c_desc_ptr = workspace + tidx * TMA_SIZE
    else:
        c_desc_ptr = None

    M_end_offset = 0
    iterated_tiles = 0

    for g in range(G):
        # For each group
        M_start_offset = M_end_offset
        m_size = tl.load(m_sizes + g)
        M_end_offset = M_start_offset + m_size

        # Only process if m_size > 0
        if m_size > 0:
            # N_start_offset is where this group's weights start in the N*G dimension
            N_start_offset = g * N

            # Calculate tiles for this group
            num_m_tiles = tl.cdiv(N, BLOCK_SIZE_M)  # Tiles along N dimension
            num_n_tiles = tl.cdiv(K, BLOCK_SIZE_N)  # Tiles along K dimension
            num_tiles = num_m_tiles * num_n_tiles

            # Setup TMA descriptor for output if using TMA
            if USE_TMA_STORE:
                tl.extra.cuda.experimental_device_tensormap_create2d(
                    desc_ptr=c_desc_ptr,
                    global_address=grad_w_ptr + N_start_offset * K,
                    load_size=[BLOCK_SIZE_M, BLOCK_SIZE_N],
                    global_size=[N, K],
                    element_ty=dtype,
                )
                tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(c_desc_ptr)

            # Process tiles
            while tidx >= iterated_tiles and tidx < iterated_tiles + num_tiles:
                gidx = tidx - iterated_tiles

                # Calculate tile indices
                tile_m_idx = gidx % num_m_tiles  # Tile index along N dimension
                tile_n_idx = gidx // num_m_tiles  # Tile index along K dimension

                # Initialize accumulator for this tile
                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

                # Calculate offsets for better memory access
                offs_m = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                offs_n = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

                # Create masks for boundary checking
                m_mask = offs_m < N
                n_mask = offs_n < K

                # Loop over the reduction dimension (M)
                for k_offset in range(0, m_size, BLOCK_SIZE_K):
                    # Handle boundary conditions for the reduction dimension
                    k_size = tl.minimum(BLOCK_SIZE_K, m_size - k_offset)
                    offs_k = tl.arange(0, BLOCK_SIZE_K)
                    k_mask = offs_k < k_size

                    # Load grad_y [M, N*G] block
                    # Shape: [BLOCK_SIZE_K, BLOCK_SIZE_M]
                    grad_y_block = tl.load(
                        grad_y_ptr
                        + (M_start_offset + k_offset + offs_k[:, None]) * (N * G)
                        + (N_start_offset + offs_m[None, :]),
                        mask=k_mask[:, None] & m_mask[None, :],
                        other=0.0,
                    )

                    # Load x_t [K, M] block
                    # Shape: [BLOCK_SIZE_N, BLOCK_SIZE_K]
                    x_t_block = tl.load(
                        x_t_ptr
                        + offs_n[:, None] * M_bucket
                        + (M_start_offset + k_offset + offs_k[None, :]),
                        mask=n_mask[:, None] & k_mask[None, :],
                        other=0.0,
                    )

                    # Matrix multiplication: (grad_y_block.T @ x_t_block)
                    accumulator += tl.dot(
                        grad_y_block.to(
                            tl.float32
                        ).T,  # Shape: [BLOCK_SIZE_M, BLOCK_SIZE_K]
                        x_t_block.to(
                            tl.float32
                        ),  # Shape: [BLOCK_SIZE_N, BLOCK_SIZE_K].T
                        allow_tf32=True,
                    )

                # Store result to grad_w
                if USE_TMA_STORE:
                    # Use TMA to store the result
                    m_offset = (tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
                    n_offset = (tile_n_idx * BLOCK_SIZE_N).to(tl.int32)

                    # Convert to output dtype and store
                    tl._experimental_descriptor_store(
                        c_desc_ptr,
                        accumulator.to(dtype),
                        [m_offset, n_offset],
                    )
                else:
                    # Manual store with boundary checking
                    tl.store(
                        grad_w_ptr
                        + (N_start_offset + offs_m[:, None]) * K
                        + offs_n[None, :],
                        accumulator.to(dtype),
                        mask=m_mask[:, None] & n_mask[None, :],
                    )

                # Move to next tile
                tidx += NUM_SMS

            # Update tiles processed counter
            iterated_tiles += num_tiles
        else:
            # Skip this group if m_size <= 0
            iterated_tiles += 0  # No tiles to process


# ======= Backwards for inputs (d_X) ======================================
@triton.jit
def _kernel_grouped_gemm_backward_x(
    grad_y_ptr,  # grad of dl/dY [M, N*G]
    w_t_ptr,  # w transposed [K, N*G]
    grad_x_ptr,  # output of kernel [M, K]
    workspace,
    m_sizes,
    # problem sizes
    G: tl.constexpr,
    M_bucket: tl.constexpr,
    N: tl.constexpr,  # N is per group
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    USE_TMA_LOAD: tl.constexpr,
    USE_TMA_STORE: tl.constexpr,
    # tile sizes - fixed for H100
    BLOCK_SIZE_M: tl.constexpr = 64,
    BLOCK_SIZE_N: tl.constexpr = 64,
    BLOCK_SIZE_K: tl.constexpr = 32,
) -> None:
    """
    Compute gradients with respect to x (input).

    grad_x = grad_y (dl/dY) @ w_t (w transposed)

    Here:
    - grad_y is [M, N*G] where N is output dim per group
    - w_t is [K, N*G] (transposed from [N*G, K])
    - grad_x is [M, K]
    """
    tidx = tl.program_id(0)
    dtype = grad_x_ptr.dtype.element_ty
    TMA_SIZE: tl.constexpr = 128

    # Initialize workspace pointer if using TMA store
    if USE_TMA_STORE:
        c_desc_ptr = workspace + tidx * TMA_SIZE
    else:
        c_desc_ptr = None

    M_end_offset = 0
    iterated_tiles = 0

    for g in range(G):
        # For each group
        M_start_offset = M_end_offset
        m_size = tl.load(m_sizes + g)
        M_end_offset = M_start_offset + m_size

        # Only process if m_size > 0
        if m_size > 0:
            # N_start_offset is where this group's weights start in the N*G dimension
            N_start_offset = g * N

            # Calculate tiles for this group
            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)  # Tiles along M dimension
            num_n_tiles = tl.cdiv(K, BLOCK_SIZE_N)  # Tiles along K dimension
            num_tiles = num_m_tiles * num_n_tiles

            # Setup TMA descriptor for output if using TMA
            if USE_TMA_STORE:
                tl.extra.cuda.experimental_device_tensormap_create2d(
                    desc_ptr=c_desc_ptr,
                    global_address=grad_x_ptr + M_start_offset * K,
                    load_size=[BLOCK_SIZE_M, BLOCK_SIZE_N],
                    global_size=[m_size, K],
                    element_ty=dtype,
                )
                tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(c_desc_ptr)

            # Process tiles
            while tidx >= iterated_tiles and tidx < iterated_tiles + num_tiles:
                gidx = tidx - iterated_tiles

                # Calculate tile indices
                tile_m_idx = gidx % num_m_tiles  # Tile index along M dimension
                tile_n_idx = gidx // num_m_tiles  # Tile index along K dimension

                # Initialize accumulator for this tile
                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

                # Calculate offsets for better memory access
                offs_m = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                offs_n = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

                # Create masks for boundary checking
                m_mask = offs_m < m_size
                n_mask = offs_n < K

                # Loop over the reduction dimension (N)
                for k_offset in range(0, N, BLOCK_SIZE_K):
                    # Handle boundary conditions for the reduction dimension
                    k_size = tl.minimum(BLOCK_SIZE_K, N - k_offset)
                    offs_k = tl.arange(0, BLOCK_SIZE_K)
                    k_mask = offs_k < k_size

                    # Load grad_y [M, N*G] block
                    # Shape: [BLOCK_SIZE_M, BLOCK_SIZE_K]
                    grad_y_block = tl.load(
                        grad_y_ptr
                        + (M_start_offset + offs_m[:, None]) * (N * G)
                        + (N_start_offset + k_offset + offs_k[None, :]),
                        mask=m_mask[:, None] & k_mask[None, :],
                        other=0.0,
                    )

                    # Load w_t [K, N*G] block
                    # Shape: [BLOCK_SIZE_N, BLOCK_SIZE_K]
                    w_t_block = tl.load(
                        w_t_ptr
                        + offs_n[:, None] * (N * G)
                        + (N_start_offset + k_offset + offs_k[None, :]),
                        mask=n_mask[:, None] & k_mask[None, :],
                        other=0.0,
                    )

                    # Matrix multiplication: grad_y @ w_t.T
                    # This computes a portion of grad_y @ w_t.T for the current tile
                    accumulator += tl.dot(
                        grad_y_block.to(tl.float32),
                        w_t_block.to(tl.float32).T,
                        allow_tf32=True,
                    )

                # Store result to grad_x
                if USE_TMA_STORE:
                    # Use TMA to store the result
                    m_offset = (tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
                    n_offset = (tile_n_idx * BLOCK_SIZE_N).to(tl.int32)

                    # Convert to output dtype and store
                    tl._experimental_descriptor_store(
                        c_desc_ptr,
                        accumulator.to(dtype),
                        [m_offset, n_offset],
                    )
                else:
                    # Manual store with boundary checking
                    tl.store(
                        grad_x_ptr
                        + (M_start_offset + offs_m[:, None]) * K
                        + offs_n[None, :],
                        accumulator.to(dtype),
                        mask=m_mask[:, None] & n_mask[None, :],
                    )

                # Move to next tile
                tidx += NUM_SMS

            # Update tiles processed counter
            iterated_tiles += num_tiles
        else:
            # Skip this group if m_size <= 0
            iterated_tiles += 0  # No tiles to process


def _compute_grad_x_pytorch(grad_output, w, m_sizes, grad_x):
    """
    Compute grad_x using pure PyTorch operations.
    """
    G = m_sizes.shape[0]
    M, _ = grad_x.shape
    N = w.shape[0] // G

    m_start = 0
    for g in range(G):
        m_size = m_sizes[g].item()
        if m_size > 0:
            m_end = m_start + m_size
            n_start = g * N
            n_end = (g + 1) * N
            grad_x[m_start:m_end] = (
                grad_output[m_start:m_end, n_start:n_end] @ w[n_start:n_end]
            )
        m_start = m_end


def _compute_grad_w_pytorch(grad_output, x, m_sizes, grad_w):
    """
    Compute grad_w using pure PyTorch operations.
    """
    G = m_sizes.shape[0]
    N = grad_w.shape[0] // G

    m_start = 0
    for g in range(G):
        m_size = m_sizes[g].item()
        if m_size > 0:
            m_end = m_start + m_size
            n_start = g * N
            n_end = (g + 1) * N
            grad_w[n_start:n_end] = (
                x[m_start:m_end].T @ grad_output[m_start:m_end, n_start:n_end]
            )
        m_start = m_end


def _pytorch_fallback_backward(grad_output, x, w, m_sizes):
    """
    Pure PyTorch implementation of grouped GEMM backward.
    """
    logging.info("Using PyTorch fallback for grouped GEMM backward")

    G = m_sizes.shape[0]
    M, K = x.shape
    N = w.shape[0] // G

    # Allocate output tensors
    grad_x = torch.zeros_like(x)
    grad_w = torch.zeros_like(w)

    # Compute grad_x and grad_w group by group
    m_start = 0
    for g in range(G):
        m_size = m_sizes[g].item()
        if m_size > 0:
            m_end = m_start + m_size
            n_start = g * N
            n_end = (g + 1) * N

            # grad_x[i] = grad_output[i] @ w
            grad_x[m_start:m_end] = (
                grad_output[m_start:m_end, n_start:n_end] @ w[n_start:n_end]
            )

            # grad_w[j] = x.T @ grad_output[j]
            grad_w[n_start:n_end] = (
                x[m_start:m_end].T @ grad_output[m_start:m_end, n_start:n_end]
            )

        m_start = m_end

    return grad_x, grad_w


def grouped_gemm_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    m_sizes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Backward pass for grouped matrix multiplication.
    Uses fixed configurations for H100 GPUs to simplify debugging.

    Args:
        grad_output: Gradient with respect to output, shape [M, N*G]
        x: Input tensor from forward pass, shape [M, K]
        w: Weight tensor from forward pass, shape [N*G, K]
        m_sizes: Group sizes tensor, shape [G]

    Returns:
        Tuple of gradients with respect to x and w: (grad_x, grad_w)
    """
    logging.info("Starting grouped_gemm_backward with fixed configurations")

    # Check CUDA availability
    if not torch.cuda.is_available():
        logging.error("CUDA not available for backward pass")
        return _pytorch_fallback_backward(grad_output, x, w, m_sizes)

    # Get GPU parameters
    device_props = torch.cuda.get_device_properties("cuda")
    if device_props.major < 9:
        logging.warning(
            "H100 or newer GPU required for optimized grouped GEMM, falling back to PyTorch"
        )
        return _pytorch_fallback_backward(grad_output, x, w, m_sizes)

    # Check TMA support
    has_tma = hasattr(tl.extra, "cuda")
    if not has_tma:
        logging.warning(
            "TMA support is required but not available, falling back to PyTorch"
        )
        return _pytorch_fallback_backward(grad_output, x, w, m_sizes)

    G = m_sizes.shape[0]
    logging.info(f"Group count: {G}")

    # Ensure contiguous tensors for efficient memory access
    grad_output = grad_output.contiguous()
    x = x.contiguous()
    w = w.contiguous()
    m_sizes = m_sizes.contiguous()

    M, K = x.shape
    N_times_G, K_w = w.shape

    logging.info(
        f"Input shapes - x: {x.shape}, w: {w.shape}, grad_output: {grad_output.shape}"
    )

    # Validate dimensions
    if K != K_w:
        logging.error(f"Input K ({K}) must match weight K ({K_w})")
        return _pytorch_fallback_backward(grad_output, x, w, m_sizes)

    if N_times_G % G != 0:
        logging.error(
            f"Weight dim 0 ({N_times_G}) must be divisible by number of groups ({G})"
        )
        return _pytorch_fallback_backward(grad_output, x, w, m_sizes)

    # Calculate N - output dimension per group
    N = N_times_G // G
    logging.info(f"N per group: {N}")

    # Verify grad_output shape
    expected_grad_output_shape = (M, N_times_G)
    if grad_output.shape != expected_grad_output_shape:
        logging.error(
            f"grad_output shape mismatch: got {grad_output.shape}, expected {expected_grad_output_shape}"
        )
        return _pytorch_fallback_backward(grad_output, x, w, m_sizes)

    # Try triton-based implementation with fallback
    try:
        # Allocate output tensors with correct shapes
        grad_x = torch.empty_like(x)
        grad_w = torch.empty_like(w)

        # Transpose x and w for backward computation
        x_t = x.T.contiguous()  # Shape: [K, M]
        w_t = w.T.contiguous()  # Shape: [K, N*G]

        # Configure kernel parameters
        NUM_SMS = device_props.multi_processor_count
        USE_TMA_LOAD = True  # Use TMA for loading
        USE_TMA_STORE = False  # Disable TMA store for better compatibility

        # Use the next power of 2 for M to avoid alignment issues
        M_bucket = triton.next_power_of_2(M)

        logging.info(f"M_bucket: {M_bucket}, NUM_SMS: {NUM_SMS}")

        # Setup TMA descriptors
        try:
            desc_helper = TmaAutoTuneHelper()

            # Create descriptors for all tensors
            desc_helper.init_tma_descriptor("grad_output")
            desc_helper.init_tma_descriptor("w_t")
            desc_helper.init_tma_descriptor("x_t")

            # Get pointers to descriptors
            grad_y_ptr = desc_helper.get_tma_descriptor_kernel_param("grad_output")
            w_t_ptr = desc_helper.get_tma_descriptor_kernel_param("w_t")
            x_t_ptr = desc_helper.get_tma_descriptor_kernel_param("x_t")

            # Allocate workspace for TMA descriptors
            workspace = torch.empty((NUM_SMS * 128), device=x.device, dtype=torch.uint8)

            # Setup TMA descriptors
            try:
                # Configure TMA descriptor for grad_output
                desc_helper.fill_2d_tma_descriptor(
                    "grad_output",
                    grad_output.data_ptr(),
                    M,
                    N_times_G,
                    64,  # BLOCK_SIZE_M
                    32,  # BLOCK_SIZE_K
                    grad_output.element_size(),
                )

                # Configure TMA descriptor for w_t
                desc_helper.fill_2d_tma_descriptor(
                    "w_t",
                    w_t.data_ptr(),
                    K,
                    N_times_G,
                    64,  # BLOCK_SIZE_N
                    32,  # BLOCK_SIZE_K
                    w_t.element_size(),
                )

                # Configure TMA descriptor for x_t
                desc_helper.fill_2d_tma_descriptor(
                    "x_t",
                    x_t.data_ptr(),
                    K,
                    M,
                    64,  # BLOCK_SIZE_N
                    32,  # BLOCK_SIZE_K
                    x_t.element_size(),
                )
            except Exception as e:
                logging.error(f"Error in TMA descriptor setup: {e}")
                return _pytorch_fallback_backward(grad_output, x, w, m_sizes)

            # Try computing grad_x using triton kernel
            try:
                logging.info("Computing grad_x with triton kernel")

                # Fixed grid size based on SM count
                # grid = (NUM_SMS,)
                grid = (min(NUM_SMS, 4),)

                _kernel_grouped_gemm_backward_x[grid](
                    grad_y_ptr,
                    w_t_ptr,
                    grad_x,
                    workspace,
                    m_sizes,
                    G,
                    M_bucket,
                    N,  # N per group
                    K,
                    NUM_SMS,
                    USE_TMA_LOAD,
                    USE_TMA_STORE,
                )
                logging.info("grad_x computation successful with triton")
            except Exception as e:
                logging.error(f"Error in backward_x kernel: {e}")
                logging.info("Falling back to PyTorch for grad_x")
                _compute_grad_x_pytorch(grad_output, w, m_sizes, grad_x)

            # Try computing grad_w using triton kernel
            try:
                logging.info("Computing grad_w with triton kernel")

                # Fixed grid size based on SM count
                # grid = (NUM_SMS,)
                grid = (min(NUM_SMS, 4),)

                _kernel_grouped_gemm_backward_w[grid](
                    x_t_ptr,
                    grad_y_ptr,
                    grad_w,
                    workspace,
                    m_sizes,
                    G,
                    M_bucket,
                    N,  # N per group
                    K,
                    NUM_SMS,
                    USE_TMA_LOAD,
                    USE_TMA_STORE,
                )
                logging.info("grad_w computation successful with triton")
            except Exception as e:
                logging.error(f"Error in backward_w kernel: {e}")
                logging.info("Falling back to PyTorch for grad_w")
                _compute_grad_w_pytorch(grad_output, x, m_sizes, grad_w)

        except Exception as e:
            logging.error(f"TMA descriptor setup failed: {e}")
            logging.info("Falling back to PyTorch implementation")
            return _pytorch_fallback_backward(grad_output, x, w, m_sizes)

        # Verify output shapes
        assert (
            grad_x.shape == x.shape
        ), f"grad_x shape mismatch: got {grad_x.shape}, expected {x.shape}"
        assert (
            grad_w.shape == w.shape
        ), f"grad_w shape mismatch: got {grad_w.shape}, expected {w.shape}"

        return grad_x, grad_w

    except Exception as e:
        logging.error(f"Unexpected error in grouped_gemm_backward: {e}")
        logging.info("Falling back to PyTorch implementation")
        return _pytorch_fallback_backward(grad_output, x, w, m_sizes)
