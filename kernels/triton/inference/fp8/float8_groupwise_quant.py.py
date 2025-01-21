# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import torch
import triton
import triton.language as tl
from triton import Config

# global constants
FP8_MAX: tl.constexpr = 448.0
EPSILON: tl.constexpr = 1e-12


@triton.jit
def _float8_groupwise_quant_kernel(
    in_ptr, out_ptr, scale_ptr, BLOCK_SIZE: tl.constexpr
):
    """
    Quantizes the input tensor via BLOCK_SIZE groupwise scaling (i.e. 1x 128).

    Results:
    Stores
    1 - float8_e4m3fn result in `out_ptr`
    2 - scaling factor in `scale_ptr`

    """
    pid = tl.program_id(axis=0)

    # load inputs
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x_vec = tl.load(in_ptr + offsets).to(tl.float32)

    # calc max and scale
    max_val = tl.max(tl.abs(x_vec))
    safe_scale = tl.maximum(max_val, EPSILON) / FP8_MAX
    y_vec = x_vec / safe_scale

    # quantize
    y_clamped = tl.minimum(tl.maximum(y_vec, -FP8_MAX), FP8_MAX)
    y_fp8 = y_clamped.to(out_ptr.dtype.element_ty)

    # store quantized values and scale
    tl.store(out_ptr + offsets, y_fp8)
    tl.store(scale_ptr + pid, safe_scale)


def float8_groupwise_quantize(x: torch.Tensor, block_size=128):
    """
    Quantizes the input tensor via block_size groupwise scaling (i.e. 1x 128)
    to torch.float8_e4m3fn format.

    Results:
    Stores
    1 - float8_e4m3fn result in `out_ptr`
    2 - scaling factor in `scale_ptr`

    """
    # verify input tensor
    x_last_dim_size = x.size(-1)

    # evenly divisible?
    if x_last_dim_size % block_size != 0:
        raise ValueError(
            f"Input tensor must have a last dimension that is a multiple of {block_size}"
        )
    # contiguous?
    if x.stride(-1) != 1:
        x = x.contiguous()

    # allocate output tensors
    output = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = x.new_empty(
        *x.size()[:-1], x_last_dim_size // block_size, dtype=torch.float32
    )
    print(f"{scales.size()=}")

    grid = lambda meta: (x.numel() // block_size,)
    _float8_groupwise_quant_kernel[grid](
        in_ptr=x,
        out_ptr=output,
        scale_ptr=scales,
        BLOCK_SIZE=block_size,
    )

    return output, scales
