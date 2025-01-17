from typing import Tuple

import torch
import triton
import triton.language as tl
from triton import Config


@triton.jit
def fast_act_quant(
    in_ptr,
    out_ptr,
    scale_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # load input block
    # mask = offsets < num_elements
    in_data = tl.load(in_ptr + offsets).to(tl.float32)

    # Find maximum / Compute scale ensuring fp8 range
    scale = tl.max(tl.abs(in_data)) / 448.0

    # Quantize values
    fp8vals = in_data / scale
    fp8vals = fp8vals.to(out_ptr.dtype.element_ty)

    # Clamp to FP8 range - do we need this?
    # fp8vals = tl.minimum(tl.maximum(fp8vals, -448.0), 448.0)

    # Store results
    tl.store(out_ptr + offsets, fp8vals)  # , mask=mask)
    # Store scale (one per block)
    tl.store(scale_ptr + pid, scale)


'''
@triton.jit
def fast_quant_kernel_128(
    x_ptr,  # input tensor
    y_ptr,  # quantized output
    s_ptr,  # scales
    n_elements,  # total elements
    BLOCK_SIZE: tl.constexpr,  # fixed at 128
):
    """
    Fast quantization kernel with fixed 128 block size.
    """
    # Get program ID
    pid = tl.program_id(axis=0)
    # Calculate block start (always aligned to 128)
    block_start = pid * BLOCK_SIZE
    # Load offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input block
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Find maximum 
    x_abs = tl.abs(x)
    max_val = tl.max(x_abs, axis=0)

    # Compute scale ensuring fp8 range
    scale = max_val / 448.0
    scale = tl.where(scale == 0, 1e-6, scale)

    # Quantize values
    x_scaled = tl.where(mask, x / scale, 0.0)
    # Use tl.cast for rounding operation
    y = tl.cast(x_scaled, tl.int8)
    y = tl.cast(y, tl.float32)

    # Clamp to FP8 range
    y = tl.minimum(tl.maximum(y, -448.0), 448.0)

    # Store results
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offsets, y, mask=mask)
    # Store scale (one per block)
    tl.store(s_ptr + pid, scale)
'''


def fast_quant_128(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optimized quantization function with fixed 128 block size.
    Args:
        x: Input tensor
    Returns:
        Tuple of (quantized tensor, scale factors)
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    n_elements = x.numel()
    block_size = 128  # Fixed block size

    assert (
        n_elements % block_size == 0
    ), "Input tensor size must be divisible by block size (128)"

    # Prepare output tensors
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    n_blocks = triton.cdiv(n_elements, block_size)
    s = torch.empty((n_blocks,), dtype=torch.float32, device=x.device)

    # Calculate grid size
    grid = (n_blocks,)

    # Launch kernel with optimal settings for 128-block
    fast_act_quant[grid](
        in_ptr=x,
        out_ptr=y,
        scale_ptr=s,
        n_elements=n_elements,
        BLOCK_SIZE=block_size,
        num_warps=8,  # Optimal for 128-size blocks
        num_stages=3,
    )

    return y, s


def verify_fast_quant_128(x: torch.Tensor, y: torch.Tensor, s: torch.Tensor) -> None:
    """Verify quantization results."""
    # Reshape tensors to match block size
    n_blocks = s.size(0)
    block_size = 128

    # Pad x if necessary to match block size
    padding_size = n_blocks * block_size - x.numel()
    if padding_size > 0:
        x_padded = torch.cat([x.flatten(), torch.zeros(padding_size, device=x.device)])
    else:
        x_padded = x.flatten()

    # Reshape for block-wise verification
    x_grouped = x_padded.reshape(-1, block_size)
    y_grouped = y.flatten().reshape(-1, block_size)

    # Calculate expected scales
    expected_s = torch.amax(torch.abs(x_grouped), dim=1) / 448.0
    expected_s = torch.where(
        expected_s == 0, torch.tensor(1e-6, device=x.device), expected_s
    )

    # Verify scales
    torch.testing.assert_close(s, expected_s, rtol=1e-3, atol=1e-3)
    print("Scales verified!")
    print(f"Expected scales: {expected_s}")
    print(f"Actual scales: {s}")

    # Verify bounds
    # error:
    # error: # RuntimeError: Promotion for Float8 Types is not supported, attempted to promote Float8_e4m3fn and Float
    # x_recon = y_grouped * s.view(-1, 1)

    # error:  RuntimeError: "abs_cuda" not implemented for 'Float8_e4m3fn'
    # assert torch.all(torch.abs(y) <= 448.0), "Values out of bounds"

    # Verify reconstruction
    # x_recon = y_grouped * s.view(-1, 1)

    # torch.testing.assert_close(x_grouped, x_recon, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    # Run tests
    x = torch.randn(2048, 1024, device="cuda")
    y, s = fast_quant_128(x)
    verify_fast_quant_128(x, y, s)
    print("Tests passed!")
