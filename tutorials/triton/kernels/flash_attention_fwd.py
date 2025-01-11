import gc
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import triton
import triton.language as tl

# Constants for configuration
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
        return False

    is_SM90 = torch.cuda.get_device_capability("cuda") >= (9, 0)
    if not is_SM90:
        print("Warning: FlashAttention with FP8 is only supported on SM90+ devices")
    return True


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


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_M": BLOCK_M,
                "BLOCK_N": BLOCK_N,
            },
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_M in BLOCK_SIZES["q"]
        for BLOCK_N in BLOCK_SIZES["kv"]
        for num_stages in NUM_STAGES
        for num_warps in NUM_WARPS
    ],
    key=["seq_len", "head_dim"],
)
@triton.jit
def _attention_kernel(
    Q,
    K,
    V,
    sm_scale,
    Out,
    Q_batch_stride,
    Q_head_stride,
    Q_seq_stride,
    Q_head_dim_stride,
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
    """Core attention kernel implementation with boundary checks."""
    start_m = tl.program_id(0)
    offset_heads = tl.program_id(1)
    offset_batch = offset_heads // num_heads
    offset_head = offset_heads % num_heads

    # Initialize pointers
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

    # Compute boundary masks for Q block
    q_block_mask = tl.arange(0, BLOCK_M) < (seq_len - start_m * BLOCK_M)

    # Load Q block with mask
    q_ptrs = (
        Q
        + q_offset
        + tl.arange(0, BLOCK_M)[:, None] * Q_seq_stride
        + tl.arange(0, head_dim)[None, :] * Q_head_dim_stride
    )
    q = tl.load(q_ptrs, mask=q_block_mask[:, None])

    # Initialize accumulator and softmax tracking
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)
    m = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Process K and V with boundary checks
    for start_n in range(0, seq_len, BLOCK_N):
        k_block_mask = tl.arange(0, BLOCK_N) < (seq_len - start_n)

        # Load K and V blocks with masks
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

        k = tl.load(k_ptrs, mask=k_block_mask[:, None])
        v = tl.load(v_ptrs, mask=k_block_mask[:, None])

        # Compute attention scores
        scores = tl.dot(q, tl.trans(k)) * sm_scale

        # Apply both causal and boundary masks
        if CAUSAL:
            scores = tl.where(
                (
                    start_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
                    >= start_n + tl.arange(0, BLOCK_N)[None, :]
                )
                & q_block_mask[:, None]
                & k_block_mask[None, :],
                scores,
                float("-inf"),
            )

        # Compute softmax with updated scores
        m_new = tl.maximum(tl.max(scores, 1), m)
        exp_scores = tl.exp(scores - m_new[:, None])  # [BLOCK_M, BLOCK_N]

        l_new = tl.sum(exp_scores, 1) + l * tl.exp(m - m_new)

        # Update accumulator

        # acc is [BLOCK_M, head_dim]

        acc_scale = tl.exp(m - m_new)  # [:, None]
        acc = acc * acc_scale[:, None]
        tmp = tl.dot(exp_scores.to(v.dtype), v)
        acc += tmp  # .to(acc.dtype)  # Add to accumulator with matching type

        # Update softmax tracking
        l = l_new
        m = m_new

    # Normalize and store output with mask
    acc = acc / l[:, None]
    out_ptrs = (
        Out
        + out_offset
        + tl.arange(0, BLOCK_M)[:, None] * O_seq_stride
        + tl.arange(0, head_dim)[None, :] * O_head_dim_stride
    )
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=q_block_mask[:, None])


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

        # Validate sequence length
        max_seq_len = BLOCK_SIZES["q"][0] * 128  # Maximum supported sequence length
        if seq_len > max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum supported length {max_seq_len}"
            )

        # Validate head dimension
        if head_dim % 8 != 0:
            raise ValueError(
                "Head dimension must be a multiple of 8 for optimal performance"
            )

        # Initialize scaling factor if needed
        if not scale_attn:
            scale_attn = 1.0 / (head_dim**0.5)

        # Prepare output tensor
        out = torch.empty_like(q)

        # Configure grid
        grid = (triton.cdiv(seq_len, BLOCK_SIZES["q"][0]), batch_size * num_heads, 1)

        # Launch kernel
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
            # BLOCK_M=BLOCK_SIZES["q"][0],
            # BLOCK_N=BLOCK_SIZES["kv"][0],
            CAUSAL=causal,
            # num_warps=4,
        )

        ctx.save_for_backward(q, k, v, out)
        ctx.causal = causal
        ctx.scale_attn = scale_attn
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        # TODO: Implement backward pass
        q, k, v, out = ctx.saved_tensors
        return None, None, None, None, None


def sram_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool = True,
    scale_attn: Optional[float] = None,
) -> torch.Tensor:
    """Wrapper around SRAMAttention forward pass"""
    check_device()

    if not all(x.is_contiguous() for x in (query, key, value)):
        raise ValueError("Input tensors must be contiguous")
    if not all(x.is_cuda for x in (query, key, value)):
        raise ValueError("Input tensors must be on CUDA device")

    return SRAMAttention.apply(query, key, value, causal, scale_attn)


def test_correctness(
    batch_size: int = 2,
    num_heads: int = 8,
    seq_len: int = 1024,
    head_dim: int = 64,
    dtype: torch.dtype = torch.bfloat16,
    atol: float = 1e-1,
    rtol: float = 1e-1,
) -> bool:
    """Test the correctness of implementation against vanilla attention."""
    try:
        torch.manual_seed(2020)
        device = torch.device("cuda")

        # Create test inputs
        q = torch.randn(
            batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        # Run both implementations
        with torch.no_grad():
            flash_output = sram_attention_forward(q, k, v, causal=True)
            vanilla_output = generic_attention(q, k, v)

        # Compare outputs
        max_diff = torch.max(torch.abs(flash_output - vanilla_output))
        mean_diff = torch.mean(torch.abs(flash_output - vanilla_output))

        print(f"Maximum difference: {max_diff:.6f}")
        print(f"Mean difference: {mean_diff:.6f}")

        return max_diff < atol and mean_diff < rtol

    except Exception as e:
        print(f"Error during correctness testing: {str(e)}")
        return False
    finally:
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    print("Running correctness test...")
    is_correct = test_correctness()
    if is_correct:
        print("✓ Implementation is correct!")
    else:
        print("✗ Implementation might have issues!")
