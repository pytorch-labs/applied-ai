# flash forward v2

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import triton
import triton.language as tl

BLOCK_SIZES = {
    "q": [64, 128],
    "kv": [32, 64],
}

NUM_STAGES = [3, 4, 7]
NUM_WARPS = [2, 4]


def check_device() -> bool:
    """Check if CUDA is available, and if the current device is a SM90+ device with FP8 support"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
        return False  # lol, I don't think this is needed

    is_SM90 = torch.cuda.get_device_capability("cuda") >= (9, 0)
    if not is_SM90:
        print("Warning: FlashAttention with FP8 is only supported on SM90+ devices")
    return True


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_M": bm,
                "BLOCK_N": bn,
            },
            num_stages=ns,
            num_warps=nw,
        )
        for bm in BLOCK_SIZES["Q"]
        for bn in BLOCK_SIZES["KV"]
        for ns in NUM_STAGES
        for nw in NUM_WARPS
    ],
    key=["seq_len", "head_dim"],
)
def generic_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale_attn: float = None
) -> torch.Tensor:
    """Generic attention kernel that works for any head_dim and seq_len."""
    if not scale_attn:
        scale_attn = 1.0 / (q.shape[-1] ** 0.5)
    # first matmul
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale_attn
    # we always use causal mask
    mask = torch.triu(torch.ones(attn.shape[-2:], device=q.device), diagonal=1).bool()
    attn.masked_fill_(mask, float("-inf"))
    # softmax
    attn = torch.softmax(attn, dim=-1)

    # second matmul
    return torch.matmul(attn, v)
