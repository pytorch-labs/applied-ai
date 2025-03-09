# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
import triton
import triton.language as tl
from tma_utils import TmaAutoTuneHelper
from triton.runtime import driver

"""
Shapes passing:
TMA benchmarks will be running with experimental grid constant TMA descriptor.
Input shape: torch.Size([1024, 256])
Output shape: torch.Size([1024, 512])
Expected output shape: [1024, 512]
Weight grad shape: torch.Size([512, 256])
Expected weight grad shape: [512, 256]
"""


# NVIDIA configurations only (FBGemm has AMD options...not yet here)
_CONFIGS = [
    triton.Config(
        {
            "BLOCK_SIZE_M": block_size_m,
            "BLOCK_SIZE_N": block_size_n,
            "BLOCK_SIZE_K": block_size_k,
        },
        num_stages=num_stages,
        num_warps=num_warps,
        num_ctas=num_ctas,
    )
    for block_size_m in [64, 128]
    for block_size_n in [64, 128, 256]
    for block_size_k in [64, 128, 256]
    for num_stages in [3, 4]
    for num_warps in [4, 8]
    for num_ctas in [1]
]


def early_config_prune(configs, name_args, **kwargs):
    device = torch.cuda.current_device()
    # dtsize = name_args["c_ptr"].element_size()
    dtsize = (
        name_args["grad_x_ptr"].element_size()
        if "grad_x_ptr" in name_args
        else name_args["grad_w_ptr"].element_size()
    )

    pruned_configs = []

    for config in configs:
        kw = config.kwargs
        BLOCK_M, BLOCK_N, BLOCK_K, num_stages = (
            kw["BLOCK_SIZE_M"],
            kw["BLOCK_SIZE_N"],
            kw["BLOCK_SIZE_K"],
            config.num_stages,
        )
        G, M, N, K = (
            name_args["G"],
            name_args["M_bucket"],
            name_args["N"],
            name_args["K"],
        )
        # verify smem
        max_shared_memory = driver.active.utils.get_device_properties(device)[
            "max_shared_mem"
        ]
        required_shared_memory = (BLOCK_M + BLOCK_N) * BLOCK_K * num_stages * dtsize
        if required_shared_memory > max_shared_memory:
            continue
        # verify M_PER_GROUP
        M_PER_GROUP = M // G
        MIN_M_TILES = 64

        # veriify we don't have excessive loading of M tiles that are too big
        if BLOCK_M > MIN_M_TILES and BLOCK_M > (M_PER_GROUP * 2):
            continue
        # don't load N tiles that are too small
        if BLOCK_M < 128 and BLOCK_M < (M_PER_GROUP // 2):
            continue

        num_sm = driver.active.utils.get_device_properties(device)[
            "multiprocessor_count"
        ]
        N_TILES = N // BLOCK_N
        MIN_N_TILES = 64
        # don't load N tiles that are too big
        if BLOCK_N > MIN_N_TILES and M * N_TILES < num_sm:
            continue
        # don't load N tilas that are too small
        if BLOCK_N < 128 and M * N_TILES < num_sm:
            continue
        # confirm K can be evenly divided
        if K % BLOCK_K != 0:
            continue

        pruned_configs.append(config)

    return pruned_configs


# ======= Backwards for d_X (inputs) ======================================
@triton.autotune(
    configs=_CONFIGS,
    key=["G", "M_bucket", "N", "K"],
    prune_configs_by={"early_config_prune": early_config_prune},
)
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
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
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
    TMA_SIZE: tl.constexpr = tl.constexpr(128)

    if USE_TMA_STORE:
        c_desc_ptr = workspace + tidx * TMA_SIZE
    else:
        c_desc_ptr = None

    M_end_offset = 0
    iterated_tiles = 0

    for g in tl.range(G):
        # For each group
        M_start_offset = M_end_offset
        m_size = tl.load(m_sizes + g)
        M_end_offset = M_start_offset + m_size

        # N_start_offset is where this group's weights start in the N*G dimension
        N_start_offset = g.to(tl.int64) * N

        is_valid_group = m_size > 0
        if is_valid_group:
            # For grad_w, we compute [N, K] for this group
            num_m_tiles = tl.cdiv(N, BLOCK_SIZE_M)  # Tiles along N dimension
            num_n_tiles = tl.cdiv(K, BLOCK_SIZE_N)  # Tiles along K dimension
            num_tiles = num_m_tiles * num_n_tiles

            if USE_TMA_STORE:
                # Set up descriptor for grad_w (output) for this group
                tl.extra.cuda.experimental_device_tensormap_create2d(
                    desc_ptr=c_desc_ptr,
                    global_address=grad_w_ptr
                    + N_start_offset * K,  # Offset to this group's weights
                    load_size=[BLOCK_SIZE_M, BLOCK_SIZE_N],
                    global_size=[N, K],  # This group's weight dimensions
                    element_ty=grad_w_ptr.dtype.element_ty,
                )
                tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(c_desc_ptr)

            # Move across tiles
            for tile_idx in range(iterated_tiles, iterated_tiles + num_tiles, NUM_SMS):
                should_process = tile_idx < iterated_tiles + num_tiles

                if should_process and (
                    (tidx < tile_idx + NUM_SMS) and (tidx >= tile_idx)
                ):
                    gidx = tidx - iterated_tiles
                    # Split tiles for output grad_w [N, K]
                    tile_m_idx = gidx % num_m_tiles  # Tile index along N dimension
                    tile_n_idx = gidx // num_m_tiles  # Tile index along K dimension

                    # Output accumulator shape: [BLOCK_SIZE_M=N_tiles, BLOCK_SIZE_N=K_tiles]
                    accumulator = tl.zeros(
                        (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32
                    )

                    # Precompute offsets for better memory access patterns
                    # N dimension offset (for grad_w outputs)
                    offs_m = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                    # K dimension offset (for grad_w outputs)
                    offs_n = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

                    # Prepare masks that will be reused
                    n_mask = offs_n < K  # K dimension boundary check
                    m_mask = offs_m < N  # N dimension boundary check

                    # Iterate over M for this group (reduction dimension)
                    for k_offset in range(0, m_size, BLOCK_SIZE_K):
                        # Compute actual K size for this block (handles edge cases)
                        k_size = tl.minimum(BLOCK_SIZE_K, m_size - k_offset)

                        # Offsets for the reduction dimension (M)
                        offs_k = tl.arange(0, BLOCK_SIZE_K)
                        k_mask = offs_k < k_size

                        # Load grad_y [M, N*G] in a more coalesced pattern
                        # Shape: [BLOCK_SIZE_K=M_tiles, BLOCK_SIZE_M=N_tiles]
                        grad_y_block = tl.load(
                            grad_y_ptr
                            # Row major access: first dimension (M) stride * N*G
                            + (M_start_offset + k_offset + offs_k[:, None]) * (N * G)
                            # Column dimension: offset to this group's N
                            + (N_start_offset + offs_m[None, :]),
                            mask=k_mask[:, None] & m_mask[None, :],
                            other=0.0,
                        )

                        # Load x_t [K, M] in a more coalesced pattern
                        # Shape: [BLOCK_SIZE_K=M_tiles, BLOCK_SIZE_N=K_tiles]
                        x_t_block = tl.load(
                            x_t_ptr
                            # K dimension is the row in x_t
                            + offs_n[None, :] * M_bucket
                            # M dimension is the column in x_t
                            + (M_start_offset + k_offset + offs_k[:, None]),
                            mask=k_mask[:, None] & n_mask[None, :],
                            other=0.0,
                        )

                        # Compute grad_w contribution: equivalent to (x_t @ grad_y)^T
                        # This is more efficient than explicit transpose operations
                        # We compute x_t.T (M,K) @ grad_y (M,N) = (K,N)
                        # Matrix dimensions: [BLOCK_SIZE_K, BLOCK_SIZE_N] @ [BLOCK_SIZE_K, BLOCK_SIZE_M] -> [BLOCK_SIZE_N, BLOCK_SIZE_M]
                        accumulator += tl.dot(
                            x_t_block.to(tl.float32).T,  # Transpose x_t_block
                            grad_y_block.to(tl.float32),
                            allow_tf32=True,
                        ).T

                    # Store the result in grad_w

                    if USE_TMA_STORE:

                        m_offset = (tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
                        n_offset = (tile_n_idx * BLOCK_SIZE_N).to(tl.int32)
                        tl._experimental_descriptor_store(
                            c_desc_ptr,
                            accumulator.to(grad_w_ptr.dtype.element_ty),
                            [m_offset, n_offset],
                        )
                    else:
                        # Store in a coalesced pattern - accessing grad_w[N*G, K]
                        # Each thread block writes BLOCK_SIZE_M x BLOCK_SIZE_N region
                        tl.store(
                            grad_w_ptr
                            # N dimension with K stride
                            + (N_start_offset + offs_m[:, None]) * K
                            # K dimension (innermost, for coalescing)
                            + offs_n[None, :],
                            accumulator.to(grad_w_ptr.dtype.element_ty),
                            mask=m_mask[:, None] & n_mask[None, :],
                        )

                tidx += NUM_SMS  # Move to next tile

            iterated_tiles += num_tiles


# Optimized kernel for grad_x computation
@triton.autotune(
    configs=_CONFIGS,
    key=["G", "M_bucket", "N", "K"],
    prune_configs_by={"early_config_prune": early_config_prune},
)
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
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
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
    TMA_SIZE: tl.constexpr = tl.constexpr(128)
    if USE_TMA_STORE:
        c_desc_ptr = workspace + tidx * TMA_SIZE
    else:
        c_desc_ptr = None

    M_end_offset = 0
    iterated_tiles = 0

    for g in tl.range(G):
        # Move across groups
        M_start_offset = M_end_offset
        m_size = tl.load(m_sizes + g)
        M_end_offset = M_start_offset + m_size

        is_valid_group = m_size > 0
        if is_valid_group:
            # N_start_offset is where this group's output starts in the N*G dimension
            N_start_offset = g.to(tl.int64) * N
            n_size = N  # This group's N size

            # For grad_x, we need to compute [M, K]
            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
            num_n_tiles = tl.cdiv(
                K, BLOCK_SIZE_N
            )  # K is output dim in backward for grad_x
            num_tiles = num_m_tiles * num_n_tiles

            if USE_TMA_STORE:
                # Set up descriptor for grad_x (output)
                tl.extra.cuda.experimental_device_tensormap_create2d(
                    desc_ptr=c_desc_ptr,
                    global_address=grad_x_ptr + M_start_offset * K,
                    load_size=[BLOCK_SIZE_M, BLOCK_SIZE_N],
                    global_size=[m_size, K],
                    element_ty=grad_x_ptr.dtype.element_ty,
                )
                tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(c_desc_ptr)

            # Move across tiles
            for tile_idx in range(iterated_tiles, iterated_tiles + num_tiles, NUM_SMS):
                should_process = tile_idx < iterated_tiles + num_tiles

                if should_process and (
                    (tidx < tile_idx + NUM_SMS) and (tidx >= tile_idx)
                ):
                    gidx = tidx - iterated_tiles
                    # Split M first and N second.
                    tile_m_idx = gidx % num_m_tiles
                    tile_n_idx = gidx // num_m_tiles

                    # Precompute offsets for better memory access patterns
                    offs_m = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                    offs_n = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

                    # Prepare masks that will be reused
                    m_mask = offs_m < m_size
                    n_mask = offs_n < K

                    accumulator = tl.zeros(
                        (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32
                    )

                    # For each K block in the inner dimension (N*G in grad_y)
                    # We iterate over the N values for this group
                    for k_offset in range(0, N, BLOCK_SIZE_K):
                        k_size = tl.minimum(BLOCK_SIZE_K, N - k_offset)
                        offs_k = tl.arange(0, BLOCK_SIZE_K)
                        k_mask = offs_k < k_size

                        # Load grad_y [M, N*G] with better coalescing
                        # Shape: [BLOCK_SIZE_M, BLOCK_SIZE_K]
                        grad_y_block = tl.load(
                            grad_y_ptr
                            # Row major: M offset with N*G stride
                            + (M_start_offset + offs_m[:, None]) * (N * G)
                            # N*G offset (column major)
                            + (N_start_offset + k_offset + offs_k[None, :]),
                            mask=m_mask[:, None] & k_mask[None, :],
                            other=0.0,
                        )

                        # Load w_t [K, N*G] with better coalescing
                        # Shape: [BLOCK_SIZE_N, BLOCK_SIZE_K]
                        w_t_block = tl.load(
                            w_t_ptr
                            # Row major: K offset with N*G stride
                            + offs_n[:, None] * (N * G)
                            # N*G offset (column major)
                            + (N_start_offset + k_offset + offs_k[None, :]),
                            mask=n_mask[:, None] & k_mask[None, :],
                            other=0.0,
                        )

                        # Compute grad_x contribution using the properly oriented matrices
                        # We need: grad_y [M,N] @ w_t.T [N,K] = [M,K]
                        accumulator += tl.dot(
                            grad_y_block.to(tl.float32),
                            w_t_block.T.to(tl.float32),
                            allow_tf32=True,
                        )

                    # Store the result in grad_x using coalesced access pattern
                    if USE_TMA_STORE:
                        m_offset = (tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
                        n_offset = (tile_n_idx * BLOCK_SIZE_N).to(tl.int32)
                        tl._experimental_descriptor_store(
                            c_desc_ptr,
                            accumulator.to(grad_x_ptr.dtype.element_ty),
                            [m_offset, n_offset],
                        )
                    else:
                        # Store grad_x [M, K] in row-major order for coalescing
                        tl.store(
                            grad_x_ptr
                            # Row major: M offset with K stride
                            + (M_start_offset + offs_m[:, None]) * K
                            # K offset (column major)
                            + offs_n[None, :],
                            accumulator.to(grad_x_ptr.dtype.element_ty),
                            mask=m_mask[:, None] & n_mask[None, :],
                        )

                tidx += NUM_SMS  # Move to next tile

            iterated_tiles += num_tiles


def grouped_gemm_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    m_sizes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Backward pass for grouped matrix multiplication.

    Args:
        grad_output: Gradient with respect to output, shape [M, N*G]
        x: Input tensor from forward pass, shape [M, K]
        w: Weight tensor from forward pass, shape [N*G, K]
        m_sizes: Group sizes tensor

    Returns:
        Tuple of gradients with respect to x and w
    """
    if not hasattr(tl.extra, "cuda"):
        raise NotImplementedError("Grouped GeMM without TMA is not available yet")

    G = m_sizes.shape[0]

    # Ensure contiguous tensors
    grad_output = grad_output.contiguous()
    x = x.contiguous()
    w = w.contiguous()
    m_sizes = m_sizes.contiguous()

    M, K = x.shape
    N_times_G = w.shape[0]

    # Calculate N - output dimension per group
    N = N_times_G // G

    assert K == w.shape[1], f"Input K ({K}) must match weight K ({w.shape[1]})"
    assert grad_output.shape == (M, N_times_G), (
        f"grad_output shape mismatch: got {grad_output.shape}, "
        f"expected ({M}, {N_times_G})"
    )

    # Transpose x and w for backward computation
    x_t = x.T.contiguous()
    w_t = w.T.contiguous()

    # Allocate output tensors with correct shapes
    grad_x = torch.empty_like(x)
    grad_w = torch.empty_like(w)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    USE_TMA_LOAD = True
    USE_TMA_STORE = True

    # Setup TMA descriptors
    desc_helper = TmaAutoTuneHelper()
    desc_helper.init_tma_descriptor("grad_output")
    desc_helper.init_tma_descriptor("w_t")
    desc_helper.init_tma_descriptor("x_t")

    grad_y_ptr = desc_helper.get_tma_descriptor_kernel_param("grad_output")
    w_t_ptr = desc_helper.get_tma_descriptor_kernel_param("w_t")
    x_t_ptr = desc_helper.get_tma_descriptor_kernel_param("x_t")

    workspace = torch.empty(
        (NUM_SMS * 128),  # TMA_SIZE
        device=x.device,
        dtype=torch.uint8,
    )

    def grid_x(META):
        # Set up TMA descriptor for grad_output with better block layout
        desc_helper.fill_2d_tma_descriptor(
            "grad_output",
            grad_output.data_ptr(),
            M,
            N_times_G,  # Full N*G dimension
            META["BLOCK_SIZE_M"],
            META["BLOCK_SIZE_K"],
            grad_output.element_size(),
        )

        # Set up TMA descriptor for w_t with better block layout
        desc_helper.fill_2d_tma_descriptor(
            "w_t",
            w_t.data_ptr(),
            K,
            N_times_G,  # Full N*G dimension
            META["BLOCK_SIZE_N"],
            META["BLOCK_SIZE_K"],
            w_t.element_size(),
        )
        return (NUM_SMS,)

    def grid_w(META):
        # Set up TMA descriptor for x_t with better block layout
        desc_helper.fill_2d_tma_descriptor(
            "x_t",
            x_t.data_ptr(),
            K,
            M,
            META["BLOCK_SIZE_N"],
            META["BLOCK_SIZE_K"],
            x_t.element_size(),
        )

        # Set up TMA descriptor for grad_output with better block layout
        desc_helper.fill_2d_tma_descriptor(
            "grad_output",
            grad_output.data_ptr(),
            M,
            N_times_G,  # Full N*G dimension
            META["BLOCK_SIZE_K"],
            META["BLOCK_SIZE_M"],
            grad_output.element_size(),
        )
        return (NUM_SMS,)

    # Launch kernels with the next power of 2 for M
    M_bucket = triton.next_power_of_2(M)

    # Compute grad_x
    _kernel_grouped_gemm_backward_x[grid_x](
        grad_y_ptr,
        w_t_ptr,
        grad_x,
        workspace,
        m_sizes,
        G,
        M_bucket,
        N,  # Pass N (per group) to kernel
        K,
        NUM_SMS,
        USE_TMA_LOAD,
        USE_TMA_STORE,
    )

    # Compute grad_w
    _kernel_grouped_gemm_backward_w[grid_w](
        x_t_ptr,
        grad_y_ptr,
        grad_w,
        workspace,
        m_sizes,
        G,
        M_bucket,
        N,  # Pass N (per group) to kernel
        K,
        NUM_SMS,
        USE_TMA_LOAD,
        USE_TMA_STORE,
    )

    return grad_x, grad_w
