# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
import triton
import triton.language as tl
from triton.runtime import driver
from tma_utils import TmaAutoTuneHelper

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
    dtsize = name_args["c_ptr"].element_size()

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
            name_args["M_BUCKET"],
            name_args["N"],
            name_args["K"],
        )
        # verify smem
        max_shared_memory = driver.get_device_properties(device)["max_shared_mem"]
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
    key=["G", "M_BUCKET", "N", "K"],
    prune_configs_by={"early_config_prune": early_config_prune},
)
@triton.jit
def _kernel_grouped_gemm_backward_x(
    grad_y_ptr,  # grad of dl/dY
    w_t_ptr,  # w transposed
    grad_x_ptr,  # output of kernel
    workspace,
    m_sizes,
    # problem sizes
    G: tl.constexpr,
    M_BUCKET: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    USE_TMA_LOAD: tl.constexpr,
    USE_TMA_STORE: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexper,
    workspace,
    m_sizes,
    # problem sizes
    G: tl.constexpr,
    M_BUCKET: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    USE_TMA_LOAD: tl.constexpr,
    USE_TMA_STORE: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N,
) -> None:
    """
    Compute gradients with respect to x (input).

    grad_x = grad_y (dl/dY) @ w_t (w transposed)

    Here:
    - grad_y is [M, N]
    - w_t is [N*G, K] transposed to [K, N*G]
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

        if m_size <= 0:
            continue

        N_start_offset = g.to(tl.int64) * N
        n_size = N

        num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(K, BLOCK_SIZE_N)  # K is output dim in backward
        num_tiles = num_m_tiles * num_n_tiles

        # TODO - get things working with TMA first, then try to get it working without TMA
        tl.extra.cuda.experimental_device_tensormap_create2d(
            desc_ptr=c_desc_ptr,
            global_address=grad_x_ptr + M_start_offset * K,
            load_size=[BLOCK_SIZE_M, BLOCK_SIZE_N],
            global_size=[m_size, K],
            element_ty=grad_x_ptr.dtype.element_ty,
        )
        tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(c_desc_ptr)

        # move across tiles
        while tidx >= iterated_tiles and tidx < iterated_tiles + num_tiles:
            gidx = tidx - iterated_tiles
            # split M first and N second.
            tile_m_idx = gidx % num_m_tiles
            tile_n_idx = gidx // num_m_tiles

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

            # For each K block in the inner dimension (N in forward pass)
            for k_offset in range(0, N, BLOCK_SIZE_K):
                k_size = tl.minimum(BLOCK_SIZE_K, N - k_offset)

                # Load grad_y block [M, K]
                offs_m = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                offs_k = tl.arange(0, BLOCK_SIZE_K)

                # We need to check if offs_m is within bounds
                m_mask = offs_m < m_size
                k_mask = offs_k < k_size

                # Load grad_y [M, N]
                grad_y_block = tl.load(
                    grad_y_ptr
                    + (M_start_offset + offs_m[:, None]) * N
                    + (N_start_offset + k_offset + offs_k[None, :]),
                    mask=m_mask[:, None] & k_mask[None, :],
                    other=0.0,
                )

                # Load w_t [K, N]
                offs_n = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                w_t_block = tl.load(
                    w_t_ptr
                    + (N_start_offset + k_offset + offs_k[:, None]) * K
                    + offs_n[None, :],
                    mask=k_mask[:, None] & (offs_n[None, :] < K),
                    other=0.0,
                )

                # Compute grad_x contribution from this block
                accumulator += tl.dot(grad_y_block, w_t_block)

            # Store the result in grad_x
            offs_m = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_n = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

            # TODO - get things working with TMA first, then try to get it working without TMA
            m_offset = (tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
            n_offset = (tile_n_idx * BLOCK_SIZE_N).to(tl.int32)
            tl.experimental_descriptor_store(
                c_desc_ptr,
                accumulator.to(grad_x_ptr.dtype.element_ty),
                [m_offset, n_offset],
            )
            tidx += NUM_SMS  # move to next group

        iterated_tiles += num_tiles
        # end kernel


# ======= Backwards for d_W (weights) ======================================
@triton.autotune(
    configs=_CONFIGS,
    key=["G", "M_BUCKET", "N", "K"],
    prune_configs_by={"early_config_prune": early_config_prune},
)

@triton.jit
def _kernel_grouped_gemm_backward_w(
    x_t_ptr,  # x transposed
    grad_y_ptr,  # grad of dl/dY
    grad_w_ptr,  # output of kernel (grad_w)
    workspace,
    m_sizes,
    # problem sizes
    G: tl.constexpr,
    M_BUCKET: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    USE_TMA_LOAD: tl.constexpr,
    USE_TMA_STORE: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K
    workspace,
    m_sizes,
    # problem sizes
    G: tl.constexpr,
    M_BUCKET: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    USE_TMA_LOAD: tl.constexpr,
    USE_TMA_STORE: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    )-> None:
    """
        Compute gradients with respect to w (weights).

        grad_w = x_t (x transposed) @ grad_y (dl/dY)

        Here:
        - x_t is [M_K] but transposed to [K, M]
        - grad_y is [M, N]
        - grad_w is [N*G, K]
    """
    tidx = tl.program_id(0)

    dtype = grad_w_ptr.dtype.element_ty

    TMA_SIZE: tl.constexpr = tl.constexpr(128)
    c_desc_ptr = workspace + tidx * TMA_SIZE

    iterated_tiles = 0
    for g in tl.range(G):
        # For each group
        M_offsets = [0]
        for i in range(g):
            M_offsets.append(M_offsets[-1] + tl.load(m_sizes+i))

        M_start_offset = M_offsets[g]
        m_size = tl.load(m_sizes + g)
        N_start_offset = g.to(tl.int64) *N

        if m_size <=0:
            continue

        N_start_offset = g.to(tl.int64) * N

        # grad_w computes N rows of K columns for each group
        num_m_tiles = tl.cdiv(N, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(K, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles

        tl.extra.cuda.experimental_device_tensormap_create2d(
            desc_ptr = c_desc_ptr,
            global_address = grad_w_ptr + N_start_offset * K,
            load_size = [BLOCK_SIZE_M, BLOCK_SIZE_N],
            global_size = [N,K],
            element_ty = grad_w_ptr.dtype.element_ty,
        )
        tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(c_desc_ptr)

        # move across tiles
        while tidx >= iterated_tiles and tidx < iterated_tiles + num_tiles:
            gidx = tidx - iterated_tiles
            # Split M first, N Second for the output grad_w
            tile_m_idx = gidx % num_m_tiles
            tile_n_idx = gidx // num_m_tiles

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

            # for each block M in grad_y (reduction dimension)
            for k_offset in range(0, m_size, BLOCK_SIZE_K):
                m_size = tl.minimum(BLOCK_SIZE_K, m_size - k_offset)

                # load x_t [K, M]
                offs_k = tl.arange(0, BLOCK_SIZE_K)
                offs_n = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

                k_mask = offs_k < k_size
                n_mask = offs_n < K

                x_t_block = tl.load(
                    x_t_ptr
                    + offs_n[None, :] * M
                    + (M_start_offset + k_offset + offs_k[:, None]),
                    mask=k_mask[:, None] & n_mask[None, :],
                    other=0.0,
                )

                # load grad_y [M, N]
                offs_m = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                m_mask = offs_m < N

                grad_y_block = tl.load(
                    grad_y_ptr
                    + (M_start_offset + k_offset + offs_k[:, None]) * N
                    + (N_start_offset + offs_m[None, :]),
                    mask=k_mask[:, None] & m_mask[None, :],
                    other=0.0,
                )

                # compute grad_w contribution from this block
                accumulator += tl.dot(x_t_block.T, grad_y_block.T)
            # store result to grad_w
            offs_m = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_n = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

            m_offset = (tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
            n_offset = (tile_n_idx * BLOCK_SIZE_N).to(tl.int32)

            tl.experimental_descriptor_store(
                c_desc_ptr,
                accumulator.to(grad_w_ptr.dtype.element_ty),
                [m_offset, n_offset],
            )
            tidx += NUM_SMS  # move to next group

        iterated_tiles += num_tiles
        # end kernel

def grouped_gemm_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    m_sizes: torch.Tensor,
    # x_scale: Optional[torch.Tensor] = None,
    # w_scale: Optional[torch.Tensor] = None,
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
        raise NotImplementedError("Grouped GeMM without TMA is not available ...yet")

    G = m_sizes.shape[0]

    # Ensure contiguous tensors
    grad_output = grad_output.contiguous()
    x = x.contiguous()
    w = w.contiguous()
    m_sizes = m_sizes.contiguous()

    M, K = x.shape
    N = w.shape[0] // G
    assert K == w.shape[1]

    # Transpose x and w
    x_t = x.T
    w_t = w.T

    # Allocate output tensors
    grad_x = torch.empty_like(x)
    grad_w = torch.empty_like(w)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    USE_TMA_LOAD = True
    USE_TMA_STORE = True

    workspace = None

    # Setup TMA descriptors
    desc_helper = utils.TmaAutoTuneHelper()
    desc_helper.init_tma_descriptor("grad_output")
    desc_helper.init_tma_descriptor("w_t")
    desc_helper.init_tma_descriptor("x_t")

    grad_y_ptr = desc_helper.get_tma_descriptor_kernel_param("grad_output")
    w_t_ptr = desc_helper.get_tma_descriptor_kernel_param("w_t")
    x_t_ptr = desc_helper.get_tma_descriptor_kernel_param("grad_output")

    workspace = torch.empty(
        (NUM_SMS * 128), # TMA_SIZE
        device=x.device,
        dtype=torch.uint8,
    )

    def grid_x(META):

        desc_helper.fill_2d_tma_descriptor(
            "grad_output",
            grad_output.data_ptr(),
            M,
            N*G,
            META["BLOCK_SIZE_M"],
            META["BLOCK_SIZE_K"],
            grad_output.element_size(),
        )



        desc_helper.fill_2d_tma_descriptor(
            "w_t",
            w_t.data_ptr(),
            K,
            N*G,
            META["BLOCK_SIZE_N"],
            META["BLOCK_SIZE_K"],
            w_t.element_size(),
        )
        return (NUM_SMS,)

    def grid_w(META):

        # grad_w computation
        desc_helper.fill_2d_tma_descriptor(
            "x_t",
            x_t.data_ptr(),
            K,
            M,
            META["BLOCK_SIZE_N"],
            META["BLOCK_SIZE_K"],
            x_t.element_size(),
        )

        desc_helper.fill_2d_tma_descriptor(
            "grad_output",
            grad_output.data_ptr(),
            M,
            N*G,
            META["BLOCK_SIZE_K"],
            META["BLOCK_SIZE_M"],
            grad_output.element_size(),
        )
        return (NUM_SMS,)

    # Launch kernels
    M_bucket = triton.next_power_of_2(M)

    # Compute grad_x
    _kernel_grouped_gemm_backward_x[grid_x](
        grad_y_ptr,
        w_t_ptr,
        grad_x,
        workspace,
        m_sizes,
        G,
        M_BUCKET,
        N,
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
        M_BUCKET,
        N,
        K,
        NUM_SMS,
        USE_TMA_LOAD,
        USE_TMA_STORE,
    )

    return grad_x, grad_w
