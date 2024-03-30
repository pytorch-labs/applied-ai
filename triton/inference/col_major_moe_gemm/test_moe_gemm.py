# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from vllm.model_executor.layers.fused_moe import fused_moe
from vllm.model_executor.layers.activation import SiluAndMul
from v0_moe_fused import fused_moe as fused_moe_v0
from v1_moe_fused import fused_moe as fused_moe_v1
from splitk_moe_fused import fused_moe
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


@pytest.mark.parametrize("m", [2, 4, 8, 16, 32, 64, 128, 512, 1024, 2048])
@pytest.mark.parametrize("n", [14336//2])
@pytest.mark.parametrize("k", [4096])
@pytest.mark.parametrize("e", [8])
@pytest.mark.parametrize("topk", [2])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_fused_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
):

    torch.cuda.manual_seed(3227)
    a = torch.randn((m, k), device='cuda', dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device='cuda', dtype=dtype) / 10

    w2 = torch.randn((e, k, n), device='cuda', dtype=dtype) / 10

    score = torch.randn((m, e), device='cuda', dtype=dtype)
    score = torch.softmax(score, dim=-1)

    topk_weight, topk_ids = torch.topk(score, topk)

    start = time.time()
    triton_output_gl = fused_moe_v0(a, w1, w2, topk_weight, topk_ids, False)
    end = time.time()

    gl_time = end - start
    gl_time = gl_time * 1000
    print("Grouped Launch Time (us): \n", gl_time)


    start = time.time()
    triton_output_cm = fused_moe_v1(a, w1, w2, topk_weight, topk_ids, False)
    end = time.time()
    cm_major_time = end - start
    cm_major_time = cm_major_time * 1000
    print("Columm Major Time (us): \n", cm_major_time)


    torch_base = torch_moe(a, w1, w2, topk_weight, topk_ids)

    assert torch.allclose(triton_output_cm, torch_base, atol=1e-2, rtol=0)
    assert torch.allclose(triton_output_cm, triton_output_gl, atol=1e-2, rtol=0)

    # print(f"{triton_output_cm=}\n")
    # print(f"{triton_output_gl=}\n")
    # print(f"{torch_base=}\n")

    print(f"Col Major Speedup: {((gl_time/cm_major_time))} x\n")
