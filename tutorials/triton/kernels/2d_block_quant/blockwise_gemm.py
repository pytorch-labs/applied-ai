import triton
import triton.language as tl
import triton.tools.experimental_descriptor
import numpy as np
import torch

@triton.jit
def gemm_kernel_tma_blockwise_persistent(
                    a_desc_ptr, 
                    b_desc_ptr, 
                    c_desc_ptr,
                    scale_a_ptr,
                    scale_b_ptr, 
                    stride_am_scale,
                    stride_ak_scale,
                    stride_bk_scale,
                    stride_bn_scale,
                    M,
                    N,
                    K,
                    BLOCK_M: tl.constexpr, 
                    BLOCK_N: tl.constexpr, 
                    BLOCK_K: tl.constexpr,
                    SCALE_BLOCK_M: tl.constexpr,
                    SCALE_BLOCK_N: tl.constexpr,
                    SCALE_BLOCK_K: tl.constexpr,
                    NUM_SM: tl.constexpr
                    ):
    
    pid_sm = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n

    tiles_per_SM = num_tiles // NUM_SM
    if pid_sm < num_tiles % NUM_SM:
        tiles_per_SM += 1

    tile_id = pid_sm - NUM_SM
    ki = -1

    pid_m = 0
    pid_n = 0
    offs_am = 0
    offs_bn = 0

    offs_scale_m = 0 
    offs_scale_n = 0
    offs_scale_k = 0

    scale_a_ptrs = scale_a_ptr
    scale_b_ptrs = scale_b_ptr

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:

            tile_id += NUM_SM

            pid_m = tile_id // num_pid_m
            pid_n = tile_id % num_pid_m

            offs_am = pid_m * BLOCK_M
            offs_bn = pid_n * BLOCK_N

            offs_am = tl.multiple_of(offs_am, BLOCK_M)
            offs_bn = tl.multiple_of(offs_bn, BLOCK_N)

            offs_scale_m = (pid_m * BLOCK_M) // SCALE_BLOCK_M
            offs_scale_n = (pid_n * BLOCK_N) // SCALE_BLOCK_N
            offs_scale_k = BLOCK_K // SCALE_BLOCK_K 

            scale_a_ptrs = scale_a_ptr + offs_scale_m*stride_am_scale + offs_scale_k*stride_ak_scale
            scale_b_ptrs = scale_b_ptr + offs_scale_k*stride_bk_scale + offs_scale_n*stride_bn_scale

        offs_k = ki * BLOCK_K
        offs_scale_k = ki * (BLOCK_K // SCALE_BLOCK_K)

        scale_a = tl.load(scale_a_ptrs)
        scale_b = tl.load(scale_b_ptrs)

        a = tl._experimental_descriptor_load(a_desc_ptr, [offs_am, offs_k], [BLOCK_M, BLOCK_K], tl.float8e4nv)
        b = tl._experimental_descriptor_load(b_desc_ptr, [offs_bn, offs_k], [BLOCK_N, BLOCK_K], tl.float8e4nv)

        accumulator = tl.dot(a, b.T, acc=accumulator, out_dtype=tl.float32)
        accumulator = accumulator * scale_a * scale_b

        if ki == k_tiles - 1:
            accumulator = accumulator.to(tl.bfloat16)
            tl._experimental_descriptor_store(c_desc_ptr, accumulator, [offs_am, offs_bn])
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)


def scaled_matmul(a, b, scale_a, scale_b, config=None):

    M, _ = a.shape
    N, K = b.shape

    if config:
        BLOCK_M = config["BLOCK_M"]
        BLOCK_N = config["BLOCK_N"]
        BLOCK_K = config["BLOCK_K"]
        num_warps = config["num_warps"]
        num_stages = config["num_stages"]

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128
    SCALE_BLOCK_M = 256
    SCALE_BLOCK_K = 256
    SCALE_BLOCK_N = 256
    num_warps = 4
    num_stages = 4

    NUM_SM = torch.cuda.get_device_properties("cuda").multi_processor_count
    c = torch.empty((M, N), dtype=torch.bfloat16, device='cuda')
    desc_a = triton.tools.experimental_descriptor.create_2d_tma_descriptor(a.data_ptr(), 
                                                                           M, K, 
                                                                           BLOCK_M, BLOCK_K, 
                                                                           a.element_size())

    desc_b = triton.tools.experimental_descriptor.create_2d_tma_descriptor(b.data_ptr(), 
                                                                           N, K, 
                                                                           BLOCK_N, BLOCK_K, 
                                                                           b.element_size())

    desc_c = triton.tools.experimental_descriptor.create_2d_tma_descriptor(c.data_ptr(), 
                                                                            M, N, 
                                                                            BLOCK_M, BLOCK_N, 
                                                                            c.element_size())

    grid = (NUM_SM, 1, 1)
    k = gemm_kernel_tma_blockwise_persistent[grid](
        desc_a, 
        desc_b, 
        desc_c,
        scale_a,
        scale_b,
        scale_a.stride(0), 
        scale_a.stride(1),
        scale_b.stride(0),
        scale_b.stride(1),
        M, 
        N, 
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        SCALE_BLOCK_M,
        SCALE_BLOCK_N,
        SCALE_BLOCK_K,
        NUM_SM,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return c
