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


def _attention_kernel(
    Q,
    K,
    V,
    sm_scale,
    Out,
    Q_batch_stride,
    Q_head_stride,
    Q_seq_stride,
    Q_head_dim_stride,  # todo - this is redundant...qkv should all be same.
    K_batch_stride,
    K_head_stride,
    K_seq_stride,
    K_head_dim_stride,
    V_batch_stride,
    V_head_stride,
    V_seq_stride,
    V_head_dim_stride,
    O_batch_stride,
    O_head_stride,
    O_seq_stride,
    O_head_dim_stride,
    num_heads: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
) -> None:
    start_m = tl.program_id(0)
    offset_heads = tl.program_id(1)
    offset_batch = offset_heads // num_heads
    offset_head = offset_heads % num_heads

    # init pointers
    q_offset = (
        offset_batch * Q_batch_stride
        + offset_head * Q_head_stride
        + start_m * BLOCK_M * Q_seq_stride
    )

    k_offset = offset_batch * K_batch_stride + offset_head * K_head_stride

    v_offset = offset_batch * V_batch_stride + offset_head * V_head_stride

    out_offset = (
        offset_batch * O_batch_stride
        + offset_head * O_head_stride
        + start_m * BLOCK_M * O_seq_stride
    )

    # outer loop - load Q block
    q_ptrs = (
        Q
        + q_offset
        + tl.arange(0, BLOCK_M)[:, None] * Q_seq_stride
        + tl.arange(0, head_dim)[None, :] * Q_head_dim_stride
    )
    q = tl.load(q_ptrs)  # TODO - mask this

    # init accumulator and softmax stats
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)
    m = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Process K and V
    for start_n in (0, seq_len, BLOCK_N):
        # load

        k_ptrs = (
            K
            + k_offset
            + start_n * K_seq_stride
            + tl.arange(0, BLOCK_N)[:, None] * K_seq_stride
            + tl.arange(0, head_dim)[None, :] * K_head_dim_stride
        )
        v_ptrs = (
            V
            + v_offset
            + start_n * V_seq_stride
            + tl.arange(0, BLOCK_N)[:, None] * V_seq_stride
            + tl.arange(0, head_dim)[None, :] * V_head_dim_stride
        )

        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)

        # compute qk attn
        scores = tl.dot(q, tl.trans(k)) * sm_scale
        if CAUSAL:
            scores = tl.where(
                tl.arange(0, BLOCK_M)[None, :] < tl.arange(0, BLOCK_N)[:, None],
                scores,
                float("-inf"),
            )
        m_new = tl.maximum(m, tl.max(scores, 1))
        exp_scores = tl.exp(scores - m_new[:, None])
        l_new = tl.sum(exp_scores, 1) + l * tl.exp(m - m_new)

        # update acc
        acc_scale = tl.exp(m - m_new[:, None])
        acc = acc * acc_scale + tl.dot(exp_scores, v)

        # update stats
        l = l_new
        m = m_new

    # write back
    acc = acc / l[:, None]
    out_ptrs = (
        Out
        + out_offset
        + tl.arange(0, BLOCK_M)[:, None] * O_seq_stride
        + tl.arange(0, head_dim)[None, :] * O_head_dim_stride
    )
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty))


class SRAMAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = True,
        scale_attn: float = None,
    ) -> torch.Tensor:
        if not check_device():
            return generic_attention(q, k, v, scale_attn)

        assert all(
            x.is_contiguous() for x in (q, k, v)
        ), "Input tensors must be contiguous"
        batch_size, num_heads, seq_len, head_dim = q.shape
        assert k.shape == v.shape == (batch_size, num_heads, seq_len, head_dim)

        # init scaling factor if needed
        if not scale_attn:
            scale_attn = 1.0 / (q.shape[-1] ** 0.5)

        # output tensor
        out = torch.empty_like(q)

        # configure grid
        grid = (tl.cdiv(seq_len, BLOCK_SIZES["Q"][0]), batch_size * num_heads, 1)

        # launch kernel
        _attention_kernel[grid](
            q,
            k,
            v,
            scale_attn,
            out,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            num_heads,
            seq_len,
            head_dim,
            BLOCK_M=BLOCK_SIZES["Q"][0],
            BLOCK_N=BLOCK_SIZES["KV"][0],
            CAUSAL=causal,
            num_warps=4,
        )

        ctx.save_for_backward(q, k, v, out)
        ctx.causal = causal
        ctx.scale_attn = scale_attn
        return out
