# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Tuple, List

from tgrouped_gemm_backwards import grouped_gemm_backward

class GroupedGemmFunction(Function):
    """
    Autograd Function for Triton grouped matrix multiplication.
    """
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, group_size: int) -> torch.Tensor:
        """

        Forward pass for grouped matrix multiplication.

        Args:
            ctx: Context object for autograd
            x: Input tensor of shape [M, K]
            w: Weight tensor of shape [N*G, K]
            m_sizes: Tensor containing group sizes

        Returns:
            Output tensor of shape [M, N*G]
        """
       assert x.dim() == 2, f"Expected 2D input tensor, got {x.dim()}D"
        assert w.dim() == 2, f"Expected 2D weight tensor, got {w.dim()}D"
        assert m_sizes.dim() == 1, f"Expected 1D group sizes tensor, got {m_sizes.dim()}D"

        assert x.is_cuda, "Input tensor must be on CUDA device"
        assert w.is_cuda, "Weight tensor must be on CUDA device"
        assert m_sizes.is_cuda, "Group sizes tensor must be on CUDA device

        assert x.dtype == torch.bfloat16, f"Input tensor must be bfloat16, got {x.dtype}"
        assert w.dtype == torch.bfloat16, f"Weight tensor must be bfloat16, got {w.dtype}"

        # check tensor shapes
        M, K = x.shape
        N_times_G, K_w = w.shape
        G = m_sizes.shape[0]
        assert K == K_w, f"Input and weight dimensions must match, got {K} and {K_w}"
        assert N_times_G % G==0, f"Weight dim 0 ({N_times_G}) must be divisible by number of groups ({G})"
        # check m_sizes sum equals M
        total_m = m_sizes.sum().item()
        assert total_m == M, f"Sum of group sizes ({total_m}) must equal input dim 0 ({M})"

        # save tensors for backward pass
        ctx.save_for_backward(x, w, m_sizes)

        # perform the grouped GEMM operation
        output = grouped_gemm(x, w, m_sizes)

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None]:
        """
        Backward pass for grouped matrix multiplication.

        Args:
            ctx: Context object from forward pass
            grad_output: Gradient tensor with respect to output

        Returns:
            Tuple of gradients with respect to inputs (x, w, m_sizes)
        """
        x, w, m_sizes = ctx.saved_tensors

        # Ensure grad_output is BF16
        if grad_output.dtype != torch.bfloat16:
            grad_output = grad_output.to(torch.bfloat16)

        grad_x, grad_w = grouped_gemm_backward(grad_output, x, w, m_sizes)
        return grad_x, grad_w, None  # no gradeint for m_sizes as constant
