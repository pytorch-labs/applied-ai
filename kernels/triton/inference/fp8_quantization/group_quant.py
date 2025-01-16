from typing import Tuple

import torch
import triton
import triton.language as tl
from triton import Config


""" Auto tuning error: 
python group_quant.py 
Traceback (most recent call last):
  File "/data/users/less/local/applied-ai/kernels/triton/inference/fp8_quantization/group_quant.py", line 104, in <module>
    def act_quant_v2(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
  File "/home/less/local/miniconda3/envs/newserver/lib/python3.10/site-packages/triton/runtime/autotuner.py", line 352, in decorator
    return Autotuner(fn, fn.arg_names, configs, key, reset_to_zero, restore_value, pre_hook=pre_hook,
AttributeError: 'function' object has no attribute 'arg_names'
"""


@triton.jit
def act_quant_kernel_v2(
    x_ptr,
    y_ptr,
    s_ptr,
    stride,
    n_elements,
    GROUP_SIZE: tl.constexpr,  # Fixed at 128
    NUM_CHUNKS: tl.constexpr,  # For multi-chunk processing
):
    """
    H100-optimized quantization kernel with fixed group size of 128.
    Uses multi-chunk processing and H100-specific optimizations.

    Args:
        x_ptr: Input tensor pointer
        y_ptr: Output quantized tensor pointer
        s_ptr: Output scale tensor pointer
        stride: Stride for the input tensor
        n_elements: Total number of elements
        GROUP_SIZE: Fixed at 128 for quantization
        NUM_CHUNKS: Number of chunks to process per thread block
    """
    pid = tl.program_id(axis=0)

    # Process multiple chunks of size 128 per thread block
    chunk_size = GROUP_SIZE
    base_idx = pid * (chunk_size * NUM_CHUNKS)

    # Initialize max value accumulator
    max_val = tl.zeros([1], dtype=tl.float32)

    # Process multiple chunks
    for chunk in range(NUM_CHUNKS):
        # Calculate current chunk offset
        chunk_base = base_idx + chunk * chunk_size
        offs = chunk_base + tl.arange(0, chunk_size)

        # Create mask for bounds checking
        mask = offs < n_elements

        # Load data with improved memory coalescing
        x = tl.load(x_ptr + offs * stride, mask=mask, other=0.0)

        # Update maximum using vectorized reduction
        x_abs = tl.abs(x)
        chunk_max = tl.max(x_abs, axis=0)
        max_val = tl.maximum(max_val, chunk_max)

    # Calculate scale factor using H100's FP8 range
    s = max_val / 448.0  # Keep the original scaling factor
    s = tl.where(s == 0, 1e-10, s)  # Avoid division by zero

    # Process chunks again for quantization
    for chunk in range(NUM_CHUNKS):
        chunk_base = base_idx + chunk * chunk_size
        offs = chunk_base + tl.arange(0, chunk_size)
        mask = offs < n_elements

        # Load and quantize with improved precision
        x = tl.load(x_ptr + offs * stride, mask=mask, other=0.0)
        y = tl.where(mask, x / s, 0.0)

        # Apply stochastic rounding for better quantization
        # H100 supports efficient random number generation
        rand = tl.rand(y.shape, dtype=tl.float32)
        y = tl.floor(y + rand)

        # Store results
        y = y.to(y_ptr.dtype.element_ty)
        tl.store(y_ptr + offs * stride, y, mask=mask)

    # Store scale factor (one per group)
    group_idx = pid // NUM_CHUNKS
    if pid % NUM_CHUNKS == 0:
        tl.store(s_ptr + group_idx, s)


# Optimized configurations for H100
h100_configs = [
    Config(
        {
            "GROUP_SIZE": 128,  # Fixed as per requirement
            "NUM_CHUNKS": nc,
        },
        num_warps=w,
        num_stages=s,
    )
    for nc in [1, 2, 4]  # Process multiple chunks per block
    for w in [4, 8]  # H100 supports more warps efficiently
    for s in [3, 4]  # Pipeline stages for H100
]


@triton.autotune(configs=h100_configs, key=["n_elements"])
def act_quant_v2(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    H100-optimized activation quantization with fixed group size of 128.

    Args:
        x: Input tensor to quantize

    Returns:
        Tuple of (quantized tensor, scale factors)
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"

    # Calculate grid dimensions with fixed group size
    n_elements = x.numel()
    num_chunks = 2  # Default number of chunks (will be autotuned)
    groups = triton.cdiv(n_elements, 128 * num_chunks)

    # Prepare output tensors
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = torch.empty((groups,), dtype=torch.float32, device=x.device)

    # Launch kernel with autotuning
    act_quant_kernel_v2[(groups,)](
        x_ptr=x,
        y_ptr=y,
        s_ptr=s,
        stride=1,
        n_elements=n_elements,
        GROUP_SIZE=128,
        NUM_CHUNKS=num_chunks,
    )

    return y, s


def verify_quantization(x: torch.Tensor, y: torch.Tensor, s: torch.Tensor) -> None:
    """
    Verify the quantization results.
    """
    # Reshape input to match group size
    x_grouped = x.view(-1, 128)

    # Calculate expected scales
    expected_s = torch.amax(torch.abs(x_grouped), dim=1) / 448.0

    # Verify scales
    torch.testing.assert_close(s, expected_s, rtol=1e-3, atol=1e-3)

    # Verify quantized values are within bounds
    assert torch.all(torch.abs(y) <= 448.0), "Quantized values exceed bounds"

    # Verify reconstruction
    x_recon = y * s.view(-1, 1)
    torch.testing.assert_close(x_recon, x_grouped, rtol=1e-2, atol=1e-2)
