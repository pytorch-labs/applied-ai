import pytest
import torch
from vllm.model_executor.layers.fused_moe import fused_moe
from vllm.model_executor.layers.activation import SiluAndMul
from triton.kernels.gptq.mixtral.w4a16_fused_dequant_gemm import dequant_gemm_moe
from v0_moe_fused import fused_moe as fused_moe_base
import time

def torch_moe(a, w1, w2, topk_weight, topk_ids):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk_ids.shape[1], 1).reshape(-1, D)
    out = torch.zeros(B * topk_ids.shape[1],
                      w2.shape[1],
                      dtype=a.dtype,
                      device=a.device)
    
    topk_ids = topk_ids.view(-1)
    topk_weight = topk_weight.view(-1)
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            out[mask] = SiluAndMul()(a[mask] @ w1[i].transpose(0, 1)) @ w2[i].transpose(0, 1)
    return (out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1)).sum(dim=1)

def test_dequant_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
):  
    m = m
    n = n
    k = k
    e = e
    topk = topk
    groupsize = 128
    packed_k_dim = k // 8 
    packed_n_dim = n // 8
    g = k // groupsize
    topk = 2

    a = torch.randn((m, k), dtype=torch.float16, device='cuda')
    qw1 = torch.randint(0, 5, (e, packed_k_dim, n), device='cuda', dtype=torch.int32)
    qw2 = torch.randint(0, 5, (e, 2*n, packed_k_dim), device='cuda', dtype=torch.int32)
    qw1_zeros = torch.randint(0, 5, (e, g, packed_n_dim), device='cuda', dtype=torch.int32)
    qw2_zeros = torch.randint(0, 5, (e, g, packed_n_dim), device='cuda', dtype=torch.int32)
    qw1_scales = torch.randn((e, g, n), dtype=torch.float16, device='cuda')
    qw2_scales = torch.randn((e, g, n), dtype=torch.float16, device='cuda')
    score = torch.randn((m, e), device='cuda', dtype=torch.float16)
    score = torch.softmax(score, dim=-1)
    _, topk_ids = torch.topk(score, topk)


    # dtype = torch.float16
    # a = torch.randn((m, k), device='cuda', dtype=dtype) / 10
    # w1 = torch.randn((e, 2 * n, k), device='cuda', dtype=dtype) / 10
    # w2 = torch.randn((e, k, n), device='cuda', dtype=dtype) / 10


    # score = torch.randn((m, e), device='cuda', dtype=dtype)
    # score = torch.softmax(score, dim=-1)
    # topk_weight, topk_ids = torch.topk(score, topk)

    # triton_output_base = fused_moe_base(a, w1, w2, topk_weight, topk_ids, False)

    # print(triton_output_base)

    # breakpoint()
    c = dequant_gemm_moe(a, 
                     qw1,
                     qw2,
                     qw1_scales,
                     qw2_scales,
                     qw1_zeros,
                     qw2_zeros,
                     topk_ids,
                    )
    # print(c)
    # assert torch.allclose(triton_output_splitk, torch_output, atol=1e-1, rtol=0)

if __name__ == '__main__':

    test_dequant_moe(2, 14336//2, 4096, 8, 2)