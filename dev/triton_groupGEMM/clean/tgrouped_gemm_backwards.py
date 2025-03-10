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
def _kernel_grouped_gemm_backward_x(
    grad_output_ptr,  # grad of dl/dY [M, N*G]
    w_ptr,  # weights [N*G, K]
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
    BLOCK_SIZE_M: tl.constexpr = 64,  # Tile size for M dimension (input)
    BLOCK_SIZE_K: tl.constexpr = 64,  # Tile size for K dimension (output)
    BLOCK_SIZE_N: tl.constexpr = 32,  # Tile size for N dimension (reduction)
) -> None:
    """
    Compute gradients with respect to x (input).

    For each group g, computes: grad_x[g] = grad_output[g] @ w[g]

    Where:
    - grad_output is [M, N*G]
    - w is [N*G, K]
    - grad_x is [M, K]

    The gradient computation is done group by group, where each group has its own
    slice of the input and output tensors:
    - For group g: grad_output_g is [m_sizes[g], N]
    - For group g: w_g is [N, K]
    - For group g: grad_x_g is [m_sizes[g], K]
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
            num_m_tiles = tl.cdiv(
                m_size, BLOCK_SIZE_M
            )  # Tiles along M dimension (input)
            num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)  # Tiles along K dimension (output)
            num_tiles = num_m_tiles * num_k_tiles

            # Setup TMA descriptor for output if using TMA
            if USE_TMA_STORE:
                tl.extra.cuda.experimental_device_tensormap_create2d(
                    desc_ptr=c_desc_ptr,
                    global_address=grad_x_ptr + M_start_offset * K,
                    load_size=[BLOCK_SIZE_M, BLOCK_SIZE_K],
                    global_size=[m_size, K],
                    element_ty=dtype,
                )
                tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(c_desc_ptr)

            # Process tiles
            while tidx >= iterated_tiles and tidx < iterated_tiles + num_tiles:
                gidx = tidx - iterated_tiles

                # Calculate tile indices
                tile_m_idx = gidx % num_m_tiles  # Tile index along M dimension
                tile_k_idx = gidx // num_m_tiles  # Tile index along K dimension

                # Initialize accumulator for this tile
                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

                # Calculate offsets for better memory access
                offs_m = tile_m_idx * BLOCK_SIZE_M + tl.arange(
                    0, BLOCK_SIZE_M
                )  # M dimension
                offs_k = tile_k_idx * BLOCK_SIZE_K + tl.arange(
                    0, BLOCK_SIZE_K
                )  # K dimension

                # Create masks for boundary checking
                m_mask = offs_m < m_size
                k_mask = offs_k < K

                # Loop over the reduction dimension (N)
                for n_offset in range(0, N, BLOCK_SIZE_N):
                    # Handle boundary conditions for the reduction dimension
                    n_size = tl.minimum(BLOCK_SIZE_N, N - n_offset)
                    offs_n = tl.arange(0, BLOCK_SIZE_N)
                    n_mask = offs_n < n_size

                    # Load grad_output [M, N*G] block
                    # Shape: [BLOCK_SIZE_M, BLOCK_SIZE_N]
                    grad_output_block = tl.load(
                        grad_output_ptr
                        + (M_start_offset + offs_m[:, None]) * (N * G)
                        + (N_start_offset + n_offset + offs_n[None, :]),
                        mask=m_mask[:, None] & n_mask[None, :],
                        other=0.0,
                    )

                    # Load w [N*G, K] block
                    # Shape: [BLOCK_SIZE_N, BLOCK_SIZE_K]
                    w_block = tl.load(
                        w_ptr
                        + (N_start_offset + n_offset + offs_n[:, None]) * K
                        + offs_k[None, :],
                        mask=n_mask[:, None] & k_mask[None, :],
                        other=0.0,
                    )

                    # Matrix multiplication: grad_x = grad_output @ w
                    # For this operation we need:
                    # - grad_output_block with shape [BLOCK_SIZE_M, BLOCK_SIZE_N]
                    # - w_block with shape [BLOCK_SIZE_N, BLOCK_SIZE_K]
                    # Result: [BLOCK_SIZE_M, BLOCK_SIZE_K]
                    accumulator += tl.dot(
                        grad_output_block.to(
                            tl.float32
                        ),  # [BLOCK_SIZE_M, BLOCK_SIZE_N]
                        w_block.to(tl.float32),  # [BLOCK_SIZE_N, BLOCK_SIZE_K]
                        allow_tf32=True,
                    )

                # Store result to grad_x
                if USE_TMA_STORE:
                    # Use TMA to store the result
                    m_offset = (tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
                    k_offset = (tile_k_idx * BLOCK_SIZE_K).to(tl.int32)

                    # Convert to output dtype and store
                    tl._experimental_descriptor_store(
                        c_desc_ptr,
                        accumulator.to(dtype),
                        [m_offset, k_offset],
                    )
                else:
                    # Manual store with boundary checking
                    tl.store(
                        grad_x_ptr
                        + (M_start_offset + offs_m[:, None]) * K
                        + offs_k[None, :],
                        accumulator.to(dtype),
                        mask=m_mask[:, None] & k_mask[None, :],
                    )

                # Move to next tile
                tidx += NUM_SMS

            # Update tiles processed counter
            iterated_tiles += num_tiles


@triton.jit
def _kernel_grouped_gemm_backward_w(
    x_ptr,  # input x [M, K]
    grad_output_ptr,  # grad of dl/dY [M, N*G]
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
    BLOCK_SIZE_N: tl.constexpr = 64,  # Tile size for N dimension (output)
    BLOCK_SIZE_K: tl.constexpr = 64,  # Tile size for K dimension (output)
    BLOCK_SIZE_M: tl.constexpr = 32,  # Tile size for M dimension (reduction)
) -> None:
    """
    Compute gradients with respect to w (weights).

    For each group g, computes: grad_w[g] = grad_output[g].T @ x[g]

    Where:
    - x is [M, K]
    - grad_output is [M, N*G]
    - grad_w is [N*G, K]

    The gradient computation is done group by group, where each group has its own
    slice of the input and output tensors:
    - For group g: x_g is [m_sizes[g], K]
    - For group g: grad_output_g is [m_sizes[g], N]
    - For group g: grad_w_g is [N, K]
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
            num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)  # Tiles along N dimension (output)
            num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)  # Tiles along K dimension (output)
            num_tiles = num_n_tiles * num_k_tiles

            # Setup TMA descriptor for output if using TMA
            if USE_TMA_STORE:
                tl.extra.cuda.experimental_device_tensormap_create2d(
                    desc_ptr=c_desc_ptr,
                    global_address=grad_w_ptr + N_start_offset * K,
                    load_size=[BLOCK_SIZE_N, BLOCK_SIZE_K],
                    global_size=[N, K],
                    element_ty=dtype,
                )
                tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(c_desc_ptr)

            # Process tiles
            while tidx >= iterated_tiles and tidx < iterated_tiles + num_tiles:
                gidx = tidx - iterated_tiles

                # Calculate tile indices
                tile_n_idx = gidx % num_n_tiles  # Tile index along N dimension
                tile_k_idx = gidx // num_n_tiles  # Tile index along K dimension

                # Initialize accumulator for this tile
                accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)

                # Calculate offsets for better memory access
                offs_n = tile_n_idx * BLOCK_SIZE_N + tl.arange(
                    0, BLOCK_SIZE_N
                )  # N dimension
                offs_k = tile_k_idx * BLOCK_SIZE_K + tl.arange(
                    0, BLOCK_SIZE_K
                )  # K dimension

                # Create masks for boundary checking
                n_mask = offs_n < N
                k_mask = offs_k < K

                # Loop over the reduction dimension (M)
                for m_offset in range(0, m_size, BLOCK_SIZE_M):
                    # Handle boundary conditions for the reduction dimension
                    m_size_block = tl.minimum(BLOCK_SIZE_M, m_size - m_offset)
                    offs_m = tl.arange(0, BLOCK_SIZE_M)
                    m_mask = offs_m < m_size_block

                    # Load grad_output [M, N*G] block
                    # Shape: [BLOCK_SIZE_M, BLOCK_SIZE_N]
                    grad_output_block = tl.load(
                        grad_output_ptr
                        + (M_start_offset + m_offset + offs_m[:, None]) * (N * G)
                        + (N_start_offset + offs_n[None, :]),
                        mask=m_mask[:, None] & n_mask[None, :],
                        other=0.0,
                    )

                    # Load x [M, K] block
                    # Shape: [BLOCK_SIZE_M, BLOCK_SIZE_K]
                    x_block = tl.load(
                        x_ptr
                        + (M_start_offset + m_offset + offs_m[:, None]) * K
                        + offs_k[None, :],
                        mask=m_mask[:, None] & k_mask[None, :],
                        other=0.0,
                    )

                    # Matrix multiplication: grad_w = grad_output.T @ x
                    # For this operation we need:
                    # - grad_output_block.T with shape [BLOCK_SIZE_N, BLOCK_SIZE_M]
                    # - x_block with shape [BLOCK_SIZE_M, BLOCK_SIZE_K]
                    # Result: [BLOCK_SIZE_N, BLOCK_SIZE_K]
                    accumulator += tl.dot(
                        grad_output_block.to(
                            tl.float32
                        ).T,  # [BLOCK_SIZE_N, BLOCK_SIZE_M]
                        x_block.to(tl.float32),  # [BLOCK_SIZE_M, BLOCK_SIZE_K]
                        allow_tf32=True,
                    )

                # Store result to grad_w
                if USE_TMA_STORE:
                    # Use TMA to store the result
                    n_offset = (tile_n_idx * BLOCK_SIZE_N).to(tl.int32)
                    k_offset = (tile_k_idx * BLOCK_SIZE_K).to(tl.int32)

                    # Convert to output dtype and store
                    tl._experimental_descriptor_store(
                        c_desc_ptr,
                        accumulator.to(dtype),
                        [n_offset, k_offset],
                    )
                else:
                    # Manual store with boundary checking
                    tl.store(
                        grad_w_ptr
                        + (N_start_offset + offs_n[:, None]) * K
                        + offs_k[None, :],
                        accumulator.to(dtype),
                        mask=n_mask[:, None] & k_mask[None, :],
                    )

                # Move to next tile
                tidx += NUM_SMS

            # Update tiles processed counter
            iterated_tiles += num_tiles


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
    import logging

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

        # Configure kernel parameters
        NUM_SMS = device_props.multi_processor_count
        USE_TMA_LOAD = True  # Use TMA for loading
        USE_TMA_STORE = False  # Disable TMA store for better compatibility

        # Use the next power of 2 for M to avoid alignment issues
        M_bucket = triton.next_power_of_2(M)

        logging.info(f"M_bucket: {M_bucket}, NUM_SMS: {NUM_SMS}")

        # Allocate workspace for TMA descriptors
        workspace = torch.empty((NUM_SMS * 128), device=x.device, dtype=torch.uint8)

        # Try computing grad_x using triton kernel
        try:
            logging.info("Computing grad_x with triton kernel")

            # Fixed grid size based on SM count
            grid = (min(NUM_SMS, 4),)

            _kernel_grouped_gemm_backward_x[grid](
                grad_output,
                w,
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
            grid = (min(NUM_SMS, 4),)

            _kernel_grouped_gemm_backward_w[grid](
                x,
                grad_output,
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


def _compute_grad_x_pytorch(grad_output, w, m_sizes, grad_x):
    """
    Compute grad_x using pure PyTorch operations.

    Args:
        grad_output: Gradient with respect to output, shape [M, N*G]
        w: Weight tensor from forward pass, shape [N*G, K]
        m_sizes: Group sizes tensor, shape [G]
        grad_x: Output tensor to store results, shape [M, K]
    """
    import logging

    import torch

    G = m_sizes.shape[0]
    M, K = grad_x.shape
    N = w.shape[0] // G

    # First zero out the output to avoid accumulation issues
    grad_x.zero_()

    m_start = 0
    for g in range(G):
        m_size = m_sizes[g].item()
        if m_size > 0:
            m_end = m_start + m_size
            n_start = g * N
            n_end = (g + 1) * N

            # Get slices for this group
            grad_output_slice = grad_output[m_start:m_end, n_start:n_end]
            w_slice = w[n_start:n_end]

            # Debug info to verify shapes
            logging.debug(
                f"Group {g}: m_start={m_start}, m_end={m_end}, n_start={n_start}, n_end={n_end}"
            )
            logging.debug(
                f"grad_output_slice: {grad_output_slice.shape}, w_slice: {w_slice.shape}"
            )

            # Use higher precision for intermediate calculation
            with torch.cuda.amp.autocast(enabled=False):
                grad_output_slice = grad_output_slice.float()
                w_slice = w_slice.float()

                # Correct grad_x computation: grad_x = grad_output @ w
                # For each row i in the current slice:
                # grad_x[i] = sum_j (grad_output[i, j] * w[j])
                result = torch.matmul(grad_output_slice, w_slice)

                # Cast back to original dtype
                result = result.to(grad_x.dtype)

                # Update grad_x slice
                grad_x[m_start:m_end].copy_(result)

        m_start = m_end


def _compute_grad_w_pytorch(grad_output, x, m_sizes, grad_w):
    """
    Compute grad_w using pure PyTorch operations.

    Args:
        grad_output: Gradient with respect to output, shape [M, N*G]
        x: Input tensor from forward pass, shape [M, K]
        m_sizes: Group sizes tensor, shape [G]
        grad_w: Output tensor to store results, shape [N*G, K]
    """
    import logging

    import torch

    G = m_sizes.shape[0]
    N = grad_w.shape[0] // G

    # First zero out the output to avoid accumulation issues
    grad_w.zero_()

    m_start = 0
    for g in range(G):
        m_size = m_sizes[g].item()
        if m_size > 0:
            m_end = m_start + m_size
            n_start = g * N
            n_end = (g + 1) * N

            # Get slices for this group
            grad_output_slice = grad_output[m_start:m_end, n_start:n_end]
            x_slice = x[m_start:m_end]

            # Debug info to verify shapes
            logging.debug(
                f"Group {g}: m_start={m_start}, m_end={m_end}, n_start={n_start}, n_end={n_end}"
            )
            logging.debug(
                f"grad_output_slice: {grad_output_slice.shape}, x_slice: {x_slice.shape}"
            )

            # Use higher precision for intermediate calculation
            with torch.cuda.amp.autocast(enabled=False):
                grad_output_slice = grad_output_slice.float()
                x_slice = x_slice.float()

                # Correct grad_w computation: grad_w = grad_output.T @ x
                result = torch.matmul(grad_output_slice.t(), x_slice)

                # Cast back to original dtype
                result = result.to(grad_w.dtype)

                # Update grad_w slice
                grad_w[n_start:n_end].copy_(result)

        m_start = m_end


def _pytorch_fallback_backward(grad_output, x, w, m_sizes):
    """
    Pure PyTorch implementation of grouped GEMM backward.

    Args:
        grad_output: Gradient with respect to output, shape [M, N*G]
        x: Input tensor from forward pass, shape [M, K]
        w: Weight tensor from forward pass, shape [N*G, K]
        m_sizes: Group sizes tensor, shape [G]

    Returns:
        Tuple of gradients with respect to x and w: (grad_x, grad_w)
    """
    import logging

    import torch

    logging.info("Using PyTorch fallback for grouped GEMM backward")

    # Allocate output tensors
    grad_x = torch.zeros_like(x)
    grad_w = torch.zeros_like(w)

    # Compute grad_x and grad_w group by group
    _compute_grad_x_pytorch(grad_output, w, m_sizes, grad_x)
    _compute_grad_w_pytorch(grad_output, x, m_sizes, grad_w)

    return grad_x, grad_w
