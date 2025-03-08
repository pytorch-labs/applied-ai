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


class GroupedGemm(nn.Module):
    """
    Pytorch module for grouped matrix multiplication.
    """
    def __init__(self, num_groups: int, input_dim: int, output_dim_per_group: int):
        """
        Initialize the GroupedGemm module.

        Args:
            num_groups: Number of groups
            input_dim: Input dimension (K)
            output_dim_per_group: Output dimension per group (N)
        """
        super(GroupedGemm, self).__init__()
        self.num_groups = num_groups
        self.input_dim = input_dim
        self.output_dim_per_group = output_dim_per_group

        # Initialize weights
        self.weight = nn.Parameter(
            torch.empty(num_groups * output_dim_per_group, input_dim, dtype=torch.bfloat16)
        )

        # Initialize weights using Kaiming uniform initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor, group_sizes: List[int] = None) -> torch.Tensor:
        """
        Forward pass for the grouped GEMM module.

        Args:
            x: Input tensor of shape [M, K]
            group_sizes: List of group sizes (must sum to M)
                        If None, equal group sizes are assumed

        Returns:
            Output tensor of shape [M, N*G]
        """
        # Ensure input is BF16
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)

        # valide input dimension
        assert x.shape[1] == self.input_dim, f"Expected input dim {self.input_dim}, got {x.shape[1]}"

        # handle group sizes
        if group_sizes is None:
            # If no group sizes provided, divide evenly (with last group taking remainder)
            batch_size = x.shape[0]
            base_size = batch_size // self.num_groups
            group_sizes = [base_size] * self.num_groups
            group_sizes[-1] += batch_size - sum(group_sizes)

        # validate group sizes
        assert len(group_sizes) == self.num_groups, f"Expected {self.num_groups} groups, got {len(group_sizes)}"
        assert sum(group_sizes) == x.shape[0], f"Group sizes must sum to input dim 0 ({x.shape[0]}), got {sum(group_sizes)}"

        m_sizes = torch.tensor(group_sizes, device=x.device, dtype=torch.int32)

        # Apply grouped GEMM
        return GroupedGemmFunction.apply(x, self.weight, group_sizes)
