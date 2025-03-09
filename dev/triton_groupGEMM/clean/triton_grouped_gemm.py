# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List, Tuple

import torch
import torch.nn as nn

from tgrouped_gemm_backwards import grouped_gemm_backward
from tgrouped_gemm_forward import grouped_gemm
from torch.autograd import Function


class GroupedGemmFunction(Function):
    """
    Autograd Function for Triton grouped matrix multiplication.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w: torch.Tensor,
        bias: torch.Tensor,
        m_sizes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for grouped matrix multiplication.

        Args:
            ctx: Context object for autograd
            x: Input tensor of shape [M, K]
            w: Weight tensor of shape [N*G, K]
            bias: Bias tensor (not used in current implementation)
            m_sizes: Tensor containing group sizes

        Returns:
            Output tensor of shape [M, N*G]
        """
        assert x.dim() == 2, f"Expected 2D input tensor, got {x.dim()}D"
        assert w.dim() == 2, f"Expected 2D weight tensor, got {w.dim()}D"
        assert (
            m_sizes.dim() == 1
        ), f"Expected 1D group sizes tensor, got {m_sizes.dim()}D"

        assert x.is_cuda, "Input tensor must be on CUDA device"
        assert w.is_cuda, "Weight tensor must be on CUDA device"
        assert m_sizes.is_cuda, "Group sizes tensor must be on CUDA device"

        assert (
            x.dtype == torch.bfloat16
        ), f"Input tensor must be bfloat16, got {x.dtype}"
        assert (
            w.dtype == torch.bfloat16
        ), f"Weight tensor must be bfloat16, got {w.dtype}"

        # check tensor shapes
        M, K = x.shape
        N_times_G, K_w = w.shape
        G = m_sizes.shape[0]
        assert K == K_w, f"Input and weight dimensions must match, got {K} and {K_w}"
        assert (
            N_times_G % G == 0
        ), f"Weight dim 0 ({N_times_G}) must be divisible by number of groups ({G})"

        # Calculate N - output dimension per group
        N = N_times_G // G

        # check m_sizes sum equals M
        total_m = m_sizes.sum().item()
        assert (
            total_m == M
        ), f"Sum of group sizes ({total_m}) must equal input dim 0 ({M})"

        # save tensors for backward pass
        ctx.save_for_backward(x, w, m_sizes)

        # perform the grouped GEMM operation
        output = grouped_gemm(x, w, m_sizes)

        # Verify the output shape
        expected_output_shape = (M, N_times_G)
        assert output.shape == expected_output_shape, (
            f"Output shape mismatch: got {output.shape}, "
            f"expected {expected_output_shape}"
        )

        return output

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, None, None]:
        """
        Backward pass for grouped matrix multiplication.

        Args:
            ctx: Context object from forward pass
            grad_output: Gradient tensor with respect to output

        Returns:
            Tuple of gradients with respect to inputs (x, w, bias, m_sizes)
        """
        x, w, m_sizes = ctx.saved_tensors

        # Ensure grad_output is BF16
        if grad_output.dtype != torch.bfloat16:
            grad_output = grad_output.to(torch.bfloat16)

        # Verify grad_output shape matches forward output shape
        M, K = x.shape
        N_times_G = w.shape[0]
        expected_grad_shape = (M, N_times_G)
        assert grad_output.shape == expected_grad_shape, (
            f"grad_output shape mismatch: got {grad_output.shape}, "
            f"expected {expected_grad_shape}"
        )

        grad_x, grad_w = grouped_gemm_backward(grad_output, x, w, m_sizes)

        # Verify output gradient shapes
        assert (
            grad_x.shape == x.shape
        ), f"grad_x shape mismatch: got {grad_x.shape}, expected {x.shape}"
        assert (
            grad_w.shape == w.shape
        ), f"grad_w shape mismatch: got {grad_w.shape}, expected {w.shape}"

        return (
            grad_x,
            grad_w,
            None,
            None,
        )  # no gradient for bias and m_sizes as they are constant


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
            torch.empty(
                num_groups * output_dim_per_group, input_dim, dtype=torch.bfloat16
            )
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

        # validate input dimension
        assert (
            x.shape[1] == self.input_dim
        ), f"Expected input dim {self.input_dim}, got {x.shape[1]}"

        # handle group sizes
        if group_sizes is None:
            # If no group sizes provided, divide evenly (with last group taking remainder)
            batch_size = x.shape[0]
            base_size = batch_size // self.num_groups
            group_sizes = [base_size] * self.num_groups
            group_sizes[-1] += batch_size - sum(group_sizes)

        # validate group sizes
        assert (
            len(group_sizes) == self.num_groups
        ), f"Expected {self.num_groups} groups, got {len(group_sizes)}"
        assert (
            sum(group_sizes) == x.shape[0]
        ), f"Group sizes must sum to input dim 0 ({x.shape[0]}), got {sum(group_sizes)}"

        m_sizes = torch.tensor(group_sizes, device=x.device, dtype=torch.int32)

        # Apply grouped GEMM
        return GroupedGemmFunction.apply(x, self.weight, None, m_sizes)


# Add simple example usage
def example_usage():
    import math

    # Make sure all necessary components are imported
    try:
        from tgrouped_gemm_backwards import grouped_gemm_backward
        from tgrouped_gemm_forward import grouped_gemm
    except ImportError:
        print("Warning: Could not import required modules, this is just an example.")

    # Configuration
    batch_size = 1024
    num_groups = 4
    input_dim = 256
    output_dim_per_group = 128

    # Create equal group sizes for this example
    group_size = batch_size // num_groups
    group_sizes = [group_size] * num_groups

    # Create input tensor
    x = torch.randn(batch_size, input_dim, device="cuda").to(torch.bfloat16)

    # Create module
    grouped_gemm_module = GroupedGemm(num_groups, input_dim, output_dim_per_group)
    grouped_gemm_module.cuda()

    # Forward pass
    output = grouped_gemm_module(x, group_sizes)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: [{batch_size}, {num_groups * output_dim_per_group}]")

    # Test backward pass by computing gradients
    loss = output.sum()
    loss.backward()

    print(f"Weight grad shape: {grouped_gemm_module.weight.grad.shape}")
    print(
        f"Expected weight grad shape: [{num_groups * output_dim_per_group}, {input_dim}]"
    )


if __name__ == "__main__":
    example_usage()
