from typing import Tuple

import torch
import triton
import triton.language as tl


# Define kernel without autotuning first
@triton.jit
def group_quant_kernel_base(
    x_ptr,
    y_ptr,
    s_ptr,
    stride,
    n_elements,
    BLOCK_SIZE: tl.constexpr,  # For arange
    GROUP_SIZE: tl.constexpr,  # Fixed at 128
    NUM_CHUNKS: tl.constexpr,  # For multi-chunk processing
):
    pid = tl.program_id(axis=0)
    group_idx = pid // NUM_CHUNKS

    # Process multiple chunks of size 128 per thread block
    chunk_size = GROUP_SIZE
    base_idx = pid * (chunk_size * NUM_CHUNKS)

    # Initialize max value accumulator
    max_val = tl.zeros([1], dtype=tl.float32)

    # Create block arange once
    offs_init = tl.arange(0, BLOCK_SIZE)

    # Process multiple chunks
    for chunk in range(NUM_CHUNKS):
        chunk_base = base_idx + chunk * chunk_size
        offs = chunk_base + offs_init
        mask = offs < n_elements

        x = tl.load(x_ptr + offs * stride, mask=mask, other=0.0)
        x_abs = tl.abs(x)
        chunk_max = tl.max(x_abs, axis=0)
        max_val = tl.maximum(max_val, chunk_max)

    s = max_val / 448.0
    s = tl.where(s == 0, 1e-10, s)  # Avoid division by zero

    for chunk in range(NUM_CHUNKS):
        chunk_base = base_idx + chunk * chunk_size
        offs = chunk_base + offs_init
        mask = offs < n_elements

        x = tl.load(x_ptr + offs * stride, mask=mask, other=0.0)
        y = tl.where(mask, x / s, 0.0)
        # rand = tl.rand(y.shape)  # , dtype=tl.float32)
        # y = tl.floor(y + rand)
        y = y.to(y_ptr.dtype.element_ty)
        tl.store(y_ptr + offs * stride, y, mask=mask)

    if pid % NUM_CHUNKS == 0:
        tl.store(s_ptr + group_idx, s)


# launch the kernel with different configs
def group_quant_kernel(
    grid,
    x_ptr,
    y_ptr,
    s_ptr,
    stride,
    n_elements,
    BLOCK_SIZE: int,
    GROUP_SIZE: int,
    NUM_CHUNKS: int,
    num_warps: int,
    num_stages: int,
):
    group_quant_kernel_base[grid](
        x_ptr,
        y_ptr,
        s_ptr,
        stride,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        GROUP_SIZE=GROUP_SIZE,
        NUM_CHUNKS=NUM_CHUNKS,
        num_warps=num_warps,
        num_stages=num_stages,
    )


# Define autotuned version using heuristics
def autotune_group_quant(n_elements):
    # Heuristic autotuning based on input size
    if n_elements < 1_000_000:  # Small inputs
        return {
            "BLOCK_SIZE": 128,
            "GROUP_SIZE": 128,
            "NUM_CHUNKS": 1,
            "num_warps": 4,
            "num_stages": 3,
        }
    elif n_elements < 10_000_000:  # Medium inputs
        return {
            "BLOCK_SIZE": 128,
            "GROUP_SIZE": 128,
            "NUM_CHUNKS": 2,
            "num_warps": 8,
            "num_stages": 4,
        }
    else:  # Large inputs
        return {
            "BLOCK_SIZE": 128,
            "GROUP_SIZE": 128,
            "NUM_CHUNKS": 4,
            "num_warps": 8,
            "num_stages": 4,
        }


def act_quant_v2(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    H100-optimized activation quantization with fixed group size of 128.

    Args:
        x: Input tensor to quantize

    Returns:
        Tuple of (quantized tensor, scale factors)
    """
    if not x.is_contiguous():
        x = x.contiguous()

    # Get input size and calculate grid
    n_elements = x.numel()

    # Get tuned parameters based on input size
    params = autotune_group_quant(n_elements)

    # Calculate number of groups
    groups = triton.cdiv(n_elements, params["GROUP_SIZE"] * params["NUM_CHUNKS"])

    # Prepare output tensors
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = torch.empty((groups,), dtype=torch.float32, device=x.device)

    # Launch kernel with tuned parameters
    group_quant_kernel(
        grid=(groups,),
        x_ptr=x,
        y_ptr=y,
        s_ptr=s,
        stride=1,
        n_elements=n_elements,
        BLOCK_SIZE=params["BLOCK_SIZE"],
        GROUP_SIZE=params["GROUP_SIZE"],
        NUM_CHUNKS=params["NUM_CHUNKS"],
        num_warps=params["num_warps"],
        num_stages=params["num_stages"],
    )

    return y, s


def verify_quantization(x: torch.Tensor, y: torch.Tensor, s: torch.Tensor) -> None:
    """Verify the quantization results."""
    x_grouped = x.view(-1, 128)
    expected_s = torch.amax(torch.abs(x_grouped), dim=1) / 448.0
    torch.testing.assert_close(s, expected_s, rtol=1e-3, atol=1e-3)
    assert torch.all(torch.abs(y) <= 448.0), "Quantized values exceed bounds"
    x_recon = y * s.view(-1, 1)
    torch.testing.assert_close(x_recon, x_grouped, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    # Test the implementation
    x = torch.randn(1024, 1024, device="cuda")
    y, s = act_quant_v2(x)
    verify_quantization(x, y, s)
    print("Quantization test passed!")
