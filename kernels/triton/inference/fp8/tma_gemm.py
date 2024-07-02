import triton
import triton.language as tl
import numpy as np
import torch

@triton.jit
def gemm_kernel_tma(a_desc_ptr, b_desc_ptr, c_desc_ptr,  #
                      prob_m, prob_n, prob_k, block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr):
    
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(prob_m, block_m)
    num_pid_k = tl.cdiv(prob_k, block_k)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = pid_m * block_m
    offs_bn = pid_n * block_n
    offs_k = 0

    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
    for kk in range(0, num_pid_k):

        a = tl._experimental_descriptor_load(a_desc_ptr, [offs_am, offs_k], [block_m, block_k], tl.float8e4nv)
        b = tl._experimental_descriptor_load(b_desc_ptr, [offs_bn, offs_k], [block_n, block_k], tl.float8e4nv)
        
        accumulator = tl.dot(a, b.T, acc=accumulator, out_dtype=tl.float32)
        offs_k += block_k

    accumulator = accumulator.to(tl.float16)
    tl._experimental_descriptor_store(c_desc_ptr, accumulator, [offs_am, offs_bn])


def matmul(a, b, config=None):

    m, _ = a.shape
    k, n = b.shape

    if config:
        block_m = config["block_m"]
        block_n = config["block_n"]
        block_k = config["block_k"]
        num_warps = config["num_warps"]
        num_stages = config["num_stages"]
    
    block_m = 64
    block_n = 64
    block_k = 256
    num_warps = 4
    num_stages = 4
    TMA_SIZE = 512

    desc_a = np.empty(TMA_SIZE, dtype=np.int8)
    desc_b = np.empty(TMA_SIZE, dtype=np.int8)
    desc_c = np.empty(TMA_SIZE, dtype=np.int8)

    c = torch.empty((m, n), dtype=torch.float16, device='cuda')
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(a.data_ptr(), m, k, block_m, block_k, a.element_size(),
                                                            desc_a)
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(b.data_ptr(), n, k, block_n, block_k, b.element_size(),
                                                            desc_b)
    triton.runtime.driver.active.utils.fill_2d_tma_descriptor(c.data_ptr(), m, n, block_m, block_n, c.element_size(),
                                                            desc_c)
    desc_a = torch.tensor(desc_a, device='cuda')
    desc_b = torch.tensor(desc_b, device='cuda')
    desc_c = torch.tensor(desc_c, device='cuda')

    total_blocks_m = triton.cdiv(m, block_m)
    total_blocks_n = triton.cdiv(n, block_n)
    
    grid = (total_blocks_m * total_blocks_n, 1, 1)
    k = gemm_kernel_tma[grid](
        desc_a, desc_b, desc_c,
        m, n, k,
        block_m,
        block_n,
        block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    # with open('tma_fp8.ttgir', 'w') as f:
    #      print(k.asm['ttgir'], file=f)

    # with open('tma_fp8.ptx', 'w') as f:
    #      print(k.asm['ptx'], file=f)

    return c


if __name__ == '__main__':

    M = 128
    N = 4096
    K = 4096

    a = torch.randn((M, K), device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
    b = b.T.contiguous()

    triton = matmul(a, b)