import pytest
import torch
from vllm.model_executor.layers.fused_moe import fused_moe
from vllm.model_executor.layers.activation import SiluAndMul
from v0_moe_fused import fused_moe as fused_moe_base
from triton.kernels.mixtral.v1_moe_fused import fused_moe
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


def test_fused_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
):

    a = torch.randn((m, k), device='cuda', dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device='cuda', dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device='cuda', dtype=dtype) / 10

    score = torch.randn((m, e), device='cuda', dtype=dtype)
    score = torch.softmax(score, dim=-1)
    topk_weight, topk_ids = torch.topk(score, topk)

    triton_output_splitk = fused_moe(a, w1, w2, topk_weight, topk_ids, False)
    triton_output_base = fused_moe_base(a, w1, w2, topk_weight, topk_ids, False)


if __name__ == '__main__':

    test_fused_moe(2, 14336//2, 4096, 8, 2, torch.float16)
