# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging

# Configure logging to print to stdout
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

import os
import sys
import unittest
from typing import List, Tuple

import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# group gemm imports

if torch.cuda.is_available():
    # from fp8_gemm import quantize_fp8_row
    from groupgemm import grouped_gemm  # , grouped_gemm_fp8_rowwise
    from tgrouped_gemm_backwards import grouped_gemm_backward
    from tma_utils import HAS_TMA_DESC
else:
    raise Exception("CUDA is not available. Skipping tests.")


# Create a custom autograd Function for grouped GEMM to enable backward pass
class GroupedGEMMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, m_sizes):
        ctx.save_for_backward(x, w, m_sizes)
        return grouped_gemm(x, w, m_sizes)

    @staticmethod
    def backward(ctx, grad_output):
        x, w, m_sizes = ctx.saved_tensors
        grad_x, grad_w = grouped_gemm_backward(grad_output, x, w, m_sizes)
        return grad_x, grad_w, None


@unittest.skipIf(
    not torch.cuda.is_available()
    or torch.cuda.get_device_properties(torch.cuda.current_device()).major < 9
    or not HAS_TMA_DESC,
    "Skip when H100 or TMA is not available",
)
class TestGroupedGEMM(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_grouped_gemm_bf16(self) -> None:
        def _test_grouped_gemm_bf16(
            shape: Tuple[int, int, int, int],
            device: torch.device,
        ) -> None:
            G, M, N, K = shape
            a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
            b = torch.randn(N * G, K, dtype=torch.bfloat16, device=device)
            m_ends, _ = torch.sort(
                torch.randint(
                    low=0, high=M, size=[G - 1], device=device, dtype=torch.int32
                )
            )
            m_ends = m_ends.tolist()
            m_starts = [0] + m_ends
            m_ends = m_ends + [M]
            m_sizes = torch.tensor(
                [m_ends[i] - m_starts[i] for i in range(G)], device=device
            ).to(torch.int32)

            result = grouped_gemm(
                a,
                b,
                m_sizes,
            )
            self.assertTrue(result.shape == (M, N * G))

            expected_result = torch.zeros(M, N * G, dtype=torch.bfloat16, device=device)
            for g in range(G):
                m_start = m_starts[g]
                m_end = m_ends[g]
                n_start = g * N
                n_end = (g + 1) * N
                expected_result[m_start:m_end, n_start:n_end] = (
                    a[m_start:m_end, :] @ b[n_start:n_end, :].T
                )

            torch.testing.assert_close(result, expected_result, atol=1e-5, rtol=1.6e-2)

        for G in (1, 4, 16):
            for M in (64, 512):
                logging.info(f"Testing BF16 GMM with G={G}, M={M}")
                _test_grouped_gemm_bf16((G, M, 256, 256), torch.device("cuda"))

    def test_grouped_gemm_backward(self) -> None:
        def _test_grouped_gemm_backward(
            shape: Tuple[int, int, int, int],
            device: torch.device,
        ) -> None:
            G, M, N, K = shape

            # Create inputs that require gradients
            a = torch.randn(
                M, K, dtype=torch.bfloat16, device=device, requires_grad=True
            )
            b = torch.randn(
                N * G, K, dtype=torch.bfloat16, device=device, requires_grad=True
            )

            # Clone inputs for PyTorch reference implementation
            a_ref = a.clone().detach().requires_grad_(True)
            b_ref = b.clone().detach().requires_grad_(True)

            # Generate group sizes
            m_ends, _ = torch.sort(
                torch.randint(
                    low=0, high=M, size=[G - 1], device=device, dtype=torch.int32
                )
            )
            m_ends = m_ends.tolist()
            m_starts = [0] + m_ends
            m_ends = m_ends + [M]
            m_sizes = torch.tensor(
                [m_ends[i] - m_starts[i] for i in range(G)], device=device
            ).to(torch.int32)

            # Create a random gradient for backpropagation
            grad_output = torch.randn(M, N * G, dtype=torch.bfloat16, device=device)

            # Forward pass with our custom implementation
            result = GroupedGEMMFunction.apply(a, b, m_sizes)

            # Compute backward with our implementation
            result.backward(grad_output)

            # Compute the reference result using PyTorch operations
            expected_result = torch.zeros(M, N * G, dtype=torch.bfloat16, device=device)
            for g in range(G):
                m_start = m_starts[g]
                m_end = m_ends[g]
                n_start = g * N
                n_end = (g + 1) * N
                expected_result[m_start:m_end, n_start:n_end] = (
                    a_ref[m_start:m_end, :] @ b_ref[n_start:n_end, :].T
                )

            # Backward pass with PyTorch reference implementation
            expected_result.backward(grad_output)

            # Check forward results
            torch.testing.assert_close(result, expected_result, atol=1e-5, rtol=1.6e-2)

            # Check gradient shapes
            self.assertEqual(a.grad.shape, a_ref.grad.shape)
            self.assertEqual(b.grad.shape, b_ref.grad.shape)

            # Check gradients for input
            torch.testing.assert_close(a.grad, a_ref.grad, atol=1e-4, rtol=1e-2)

            # Check gradients for weights
            torch.testing.assert_close(b.grad, b_ref.grad, atol=1e-4, rtol=1e-2)

            # Log the maximum absolute error for debugging
            grad_a_max_error = (a.grad - a_ref.grad).abs().max().item()
            grad_b_max_error = (b.grad - b_ref.grad).abs().max().item()
            logging.info(f"Grad A max error: {grad_a_max_error}")
            logging.info(f"Grad B max error: {grad_b_max_error}")

        for G in (1, 4, 16):
            for M in (64, 512):
                logging.info(f"Testing GMM Backward with G={G}, M={M}")
                _test_grouped_gemm_backward((G, M, 256, 256), torch.device("cuda"))

    def test_grouped_gemm_backward_nontrivial_groups(self) -> None:
        """Test backward pass with specific non-trivial group sizes"""

        def _test_grouped_gemm_backward_custom_groups(
            shape: Tuple[int, int, int, int],
            group_sizes: List[int],
            device: torch.device,
        ) -> None:
            G, M, N, K = shape
            assert G == len(
                group_sizes
            ), "Number of groups must match length of group_sizes"
            assert sum(group_sizes) == M, "Total size of groups must match M dimension"

            # Create inputs that require gradients
            a = torch.randn(
                M, K, dtype=torch.bfloat16, device=device, requires_grad=True
            )
            b = torch.randn(
                N * G, K, dtype=torch.bfloat16, device=device, requires_grad=True
            )

            # Clone inputs for PyTorch reference implementation
            a_ref = a.clone().detach().requires_grad_(True)
            b_ref = b.clone().detach().requires_grad_(True)

            # Use the provided group sizes
            m_sizes = torch.tensor(group_sizes, device=device).to(torch.int32)

            # Create a random gradient for backpropagation
            grad_output = torch.randn(M, N * G, dtype=torch.bfloat16, device=device)

            # Forward pass with our custom implementation
            result = GroupedGEMMFunction.apply(a, b, m_sizes)

            # Compute backward with our implementation
            result.backward(grad_output)

            # Compute the reference result using PyTorch operations
            expected_result = torch.zeros(M, N * G, dtype=torch.bfloat16, device=device)

            m_start = 0
            for g in range(G):
                m_end = m_start + group_sizes[g]
                n_start = g * N
                n_end = (g + 1) * N
                expected_result[m_start:m_end, n_start:n_end] = (
                    a_ref[m_start:m_end, :] @ b_ref[n_start:n_end, :].T
                )
                m_start = m_end

            # Backward pass with PyTorch reference implementation
            expected_result.backward(grad_output)

            # Check forward results
            torch.testing.assert_close(result, expected_result, atol=1e-5, rtol=1.6e-2)

            # Check gradients
            torch.testing.assert_close(a.grad, a_ref.grad, atol=1e-4, rtol=1e-2)
            torch.testing.assert_close(b.grad, b_ref.grad, atol=1e-4, rtol=1e-2)

        # Test case 1: Equal group sizes
        _test_grouped_gemm_backward_custom_groups(
            (4, 128, 64, 64), [32, 32, 32, 32], torch.device("cuda")
        )

        # Test case 2: Unequal group sizes
        _test_grouped_gemm_backward_custom_groups(
            (4, 100, 64, 64), [10, 20, 30, 40], torch.device("cuda")
        )

        # Test case 3: One group is empty (size 0)
        _test_grouped_gemm_backward_custom_groups(
            (4, 90, 64, 64), [30, 0, 40, 20], torch.device("cuda")
        )

    def test_grouped_gemm_backward_numerical_gradient(self) -> None:
        """Test backward pass using numerical gradients for verification"""

        def compute_numerical_gradient(func, inputs, grad_output, eps=1e-3):
            """
            Compute numerical gradient for the given function using finite differences
            """
            numerical_grads = []
            for input_tensor in inputs:
                numerical_grad = torch.zeros_like(input_tensor)

                # Flatten the tensor for easier iteration
                flat_input = input_tensor.flatten()
                flat_grad = numerical_grad.flatten()

                # Iterate through each element
                for i in range(flat_input.numel()):
                    # Store original value
                    orig_val = flat_input[i].item()

                    # Compute f(x + eps)
                    flat_input[i] = orig_val + eps
                    output_plus = func()
                    loss_plus = (output_plus * grad_output).sum()

                    # Compute f(x - eps)
                    flat_input[i] = orig_val - eps
                    output_minus = func()
                    loss_minus = (output_minus * grad_output).sum()

                    # Restore original value
                    flat_input[i] = orig_val

                    # Compute gradient using central difference
                    flat_grad[i] = (loss_plus - loss_minus) / (2 * eps)

                numerical_grads.append(numerical_grad)

            return numerical_grads

        G, M, N, K = 2, 64, 32, 32  # Smaller dimensions for numerical gradient testing
        device = torch.device("cuda")

        # Create inputs
        a = torch.randn(M, K, dtype=torch.float32, device=device, requires_grad=True)
        b = torch.randn(
            N * G, K, dtype=torch.float32, device=device, requires_grad=True
        )

        # Create group sizes
        group_sizes = [40, 24]
        m_sizes = torch.tensor(group_sizes, device=device).to(torch.int32)

        # Create a random gradient for backpropagation
        grad_output = torch.randn(M, N * G, dtype=torch.float32, device=device)

        # Define the function to compute forward pass
        def forward_func():
            return GroupedGEMMFunction.apply(a.clone(), b.clone(), m_sizes)

        # Compute analytical gradients
        output = GroupedGEMMFunction.apply(a, b, m_sizes)
        output.backward(grad_output)
        analytical_grad_a = a.grad.clone()
        analytical_grad_b = b.grad.clone()

        # Reset gradients
        a.grad = None
        b.grad = None

        # Compute numerical gradients for a subset of elements to save computation time
        # We'll sample gradient at a few elements
        sample_size = 10

        # Sample indices for a
        a_indices = torch.randint(0, M * K, (sample_size,), device=device)

        # Sample indices for b
        b_indices = torch.randint(0, N * G * K, (sample_size,), device=device)

        # Test gradient for a at sampled locations
        for idx in a_indices:
            row, col = idx // K, idx % K

            # Manual numerical gradient computation for specific elements
            eps = 1e-3
            a[row, col] += eps
            output_plus = GroupedGEMMFunction.apply(a, b, m_sizes)
            loss_plus = (output_plus * grad_output).sum().item()

            a[row, col] -= 2 * eps  # -eps from original
            output_minus = GroupedGEMMFunction.apply(a, b, m_sizes)
            loss_minus = (output_minus * grad_output).sum().item()

            # Reset to original value
            a[row, col] += eps

            numerical_grad = (loss_plus - loss_minus) / (2 * eps)
            analytical_grad = analytical_grad_a[row, col].item()

            # Check if numerical and analytical gradients are close
            self.assertTrue(
                abs(numerical_grad - analytical_grad) < 1e-1,
                f"Gradient mismatch for a[{row},{col}]: numerical={numerical_grad}, analytical={analytical_grad}",
            )

        # Test gradient for b at sampled locations
        for idx in b_indices:
            row, col = idx // K, idx % K

            # Manual numerical gradient computation
            eps = 1e-3
            b[row, col] += eps
            output_plus = GroupedGEMMFunction.apply(a, b, m_sizes)
            loss_plus = (output_plus * grad_output).sum().item()

            b[row, col] -= 2 * eps
            output_minus = GroupedGEMMFunction.apply(a, b, m_sizes)
            loss_minus = (output_minus * grad_output).sum().item()

            # Reset to original value
            b[row, col] += eps

            numerical_grad = (loss_plus - loss_minus) / (2 * eps)
            analytical_grad = analytical_grad_b[row, col].item()

            # Check if numerical and analytical gradients are close
            self.assertTrue(
                abs(numerical_grad - analytical_grad) < 1e-1,
                f"Gradient mismatch for b[{row},{col}]: numerical={numerical_grad}, analytical={analytical_grad}",
            )

        logging.info("Numerical gradient check passed for sampled elements")


if __name__ == "__main__":
    unittest.main(exit=False)
