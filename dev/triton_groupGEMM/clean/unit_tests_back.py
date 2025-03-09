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
    from groupgemm import grouped_gemm
    from tgrouped_gemm_backwards import grouped_gemm_backward
    from tma_utils import HAS_TMA_DESC
else:
    raise Exception("CUDA is not available. Skipping tests.")


# Create a custom autograd Function for grouped GEMM to enable backward pass
class GroupedGEMMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, m_sizes):
        ctx.save_for_backward(x, w, m_sizes)
        output = grouped_gemm(x, w, m_sizes)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, w, m_sizes = ctx.saved_tensors

        # Ensure proper shapes and types
        assert (
            grad_output.dtype == x.dtype
        ), f"Grad output dtype {grad_output.dtype} doesn't match input dtype {x.dtype}"
        assert grad_output.is_contiguous(), "Grad output must be contiguous"

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

            # Create more deterministic group sizes
            m_sizes = torch.zeros(G, device=device, dtype=torch.int32)
            base_size = M // G
            remainder = M % G

            for i in range(G):
                m_sizes[i] = base_size + (1 if i < remainder else 0)

            # Verify m_sizes sum to M
            self.assertEqual(m_sizes.sum().item(), M)

            result = grouped_gemm(
                a,
                b,
                m_sizes,
            )

            # Verify output shape
            self.assertEqual(result.shape, (M, N * G))

            # Compute the expected result using PyTorch operations
            expected_result = torch.zeros(M, N * G, dtype=torch.bfloat16, device=device)
            m_start = 0
            for g in range(G):
                m_size = m_sizes[g].item()
                m_end = m_start + m_size
                n_start = g * N
                n_end = (g + 1) * N

                if m_size > 0:  # Only compute for non-empty groups
                    expected_result[m_start:m_end, n_start:n_end] = (
                        a[m_start:m_end, :] @ b[n_start:n_end, :].T
                    )

                m_start = m_end

            # Use appropriate tolerance for BF16
            torch.testing.assert_close(result, expected_result, atol=1e-2, rtol=1e-2)

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

            # Generate evenly distributed group sizes
            m_sizes = torch.zeros(G, device=device, dtype=torch.int32)
            base_size = M // G
            remainder = M % G

            for i in range(G):
                m_sizes[i] = base_size + (1 if i < remainder else 0)

            # Create correct m_starts and m_ends arrays for reference calculation
            m_starts = [0]
            for i in range(G - 1):
                m_starts.append(m_starts[-1] + m_sizes[i].item())

            m_ends = []
            for i in range(G):
                m_ends.append(m_starts[i] + m_sizes[i].item())

            # Create a random gradient for backpropagation
            grad_output = torch.randn(M, N * G, dtype=torch.bfloat16, device=device)

            # Forward pass with our custom implementation
            result = GroupedGEMMFunction.apply(a, b, m_sizes)

            # Check the shape first
            expected_shape = (M, N * G)
            self.assertEqual(
                result.shape,
                expected_shape,
                f"Forward result shape mismatch, expected {expected_shape}, got {result.shape}",
            )

            # Compute backward with our implementation
            result.backward(grad_output)

            # Compute the reference result using PyTorch operations
            expected_result = torch.zeros(M, N * G, dtype=torch.bfloat16, device=device)
            for g in range(G):
                m_start = m_starts[g]
                m_end = m_ends[g]
                n_start = g * N
                n_end = (g + 1) * N

                if m_end > m_start:  # Only compute for non-empty groups
                    expected_result[m_start:m_end, n_start:n_end] = (
                        a_ref[m_start:m_end, :] @ b_ref[n_start:n_end, :].T
                    )

            # Backward pass with PyTorch reference implementation
            expected_result.backward(grad_output)

            # Check forward results
            torch.testing.assert_close(result, expected_result, atol=1e-2, rtol=1e-2)

            # Check gradient shapes
            self.assertEqual(
                a.grad.shape,
                a_ref.grad.shape,
                f"grad_x shape mismatch, expected {a_ref.grad.shape}, got {a.grad.shape}",
            )
            self.assertEqual(
                b.grad.shape,
                b_ref.grad.shape,
                f"grad_w shape mismatch, expected {b_ref.grad.shape}, got {b.grad.shape}",
            )

            # Check gradients with appropriate tolerance for BF16
            torch.testing.assert_close(a.grad, a_ref.grad, atol=1e-2, rtol=1e-2)
            torch.testing.assert_close(b.grad, b_ref.grad, atol=1e-2, rtol=1e-2)

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

            # Create a random gradient for backpropagation that matches the expected output shape
            grad_output = torch.randn(M, N * G, dtype=torch.bfloat16, device=device)

            # Forward pass with our custom implementation
            result = GroupedGEMMFunction.apply(a, b, m_sizes)

            # Verify the shape matches expectations before backprop
            expected_shape = (M, N * G)
            self.assertEqual(
                result.shape,
                expected_shape,
                f"Forward result shape mismatch, expected {expected_shape}, got {result.shape}",
            )

            # Compute backward with our implementation
            result.backward(grad_output)

            # Compute the reference result using PyTorch operations
            expected_result = torch.zeros(M, N * G, dtype=torch.bfloat16, device=device)

            m_start = 0
            for g in range(G):
                m_end = m_start + group_sizes[g]
                n_start = g * N
                n_end = (g + 1) * N

                if group_sizes[g] > 0:  # Only compute for non-empty groups
                    expected_result[m_start:m_end, n_start:n_end] = (
                        a_ref[m_start:m_end, :] @ b_ref[n_start:n_end, :].T
                    )
                m_start = m_end

            # Backward pass with PyTorch reference implementation
            expected_result.backward(grad_output)

            # Check forward results with appropriate tolerance for BF16
            torch.testing.assert_close(result, expected_result, atol=1e-2, rtol=1e-2)

            # Check gradients with appropriate tolerance for BF16
            torch.testing.assert_close(a.grad, a_ref.grad, atol=1e-2, rtol=1e-2)
            torch.testing.assert_close(b.grad, b_ref.grad, atol=1e-2, rtol=1e-2)

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

        # Use smaller dimensions and simpler approach for numerical gradient testing
        G, M, N, K = 2, 32, 16, 16  # Smaller dimensions
        device = torch.device("cuda")

        # Create inputs
        a = torch.randn(M, K, dtype=torch.float32, device=device, requires_grad=True)
        b = torch.randn(
            N * G, K, dtype=torch.float32, device=device, requires_grad=True
        )

        # Create group sizes - keeping it simple for numerical test
        group_sizes = [M // 2, M // 2]
        m_sizes = torch.tensor(group_sizes, device=device).to(torch.int32)

        # Create a random gradient for backpropagation
        grad_output = torch.randn(M, N * G, dtype=torch.float32, device=device)

        # Convert to bfloat16 for the actual GEMM operation
        a_bf16 = a.detach().to(torch.bfloat16).requires_grad_(True)
        b_bf16 = b.detach().to(torch.bfloat16).requires_grad_(True)

        # Forward and backward with our implementation in BF16
        output_bf16 = GroupedGEMMFunction.apply(a_bf16, b_bf16, m_sizes)
        output_bf16.backward(grad_output.to(torch.bfloat16))

        # Store gradients
        analytical_grad_a = a_bf16.grad.to(torch.float32).clone()
        analytical_grad_b = b_bf16.grad.to(torch.float32).clone()

        # Clear gradients
        a_bf16.grad = None
        b_bf16.grad = None

        # Use PyTorch's autograd.gradcheck for numerical gradient verification
        # We'll need a simplified test function
        def test_function(a_input, b_input):
            # Convert to bfloat16 for GEMM operation
            a_test = a_input.to(torch.bfloat16)
            b_test = b_input.to(torch.bfloat16)

            # Compute forward
            result = GroupedGEMMFunction.apply(a_test, b_test, m_sizes)

            # Convert back to float32 for gradient check
            return result.to(torch.float32)

        # Sample a subset of elements to check (full check is too slow)
        num_samples = 5

        # Test gradient for input tensor a
        for _ in range(num_samples):
            i = torch.randint(0, M, (1,)).item()
            j = torch.randint(0, K, (1,)).item()

            # Manual finite difference approximation
            eps = 1e-3

            # Save original value
            orig_val = a[i, j].item()

            # f(x + eps)
            a[i, j] = orig_val + eps
            out1 = test_function(a, b)
            loss1 = torch.sum(out1 * grad_output).item()

            # f(x - eps)
            a[i, j] = orig_val - eps
            out2 = test_function(a, b)
            loss2 = torch.sum(out2 * grad_output).item()

            # Restore original value
            a[i, j] = orig_val

            # Compute numerical gradient
            numerical_grad = (loss1 - loss2) / (2 * eps)
            analytical_grad = analytical_grad_a[i, j].item()

            # Check if gradients are close enough
            rel_error = abs(numerical_grad - analytical_grad) / max(
                1e-8, abs(analytical_grad)
            )
            self.assertLess(
                rel_error,
                0.1,
                f"Gradient check failed for a[{i},{j}]: numerical={numerical_grad}, analytical={analytical_grad}",
            )

        # Test gradient for weight tensor b (similarly)
        for _ in range(num_samples):
            i = torch.randint(0, N * G, (1,)).item()
            j = torch.randint(0, K, (1,)).item()

            # Manual finite difference approximation
            eps = 1e-3

            # Save original value
            orig_val = b[i, j].item()

            # f(x + eps)
            b[i, j] = orig_val + eps
            out1 = test_function(a, b)
            loss1 = torch.sum(out1 * grad_output).item()

            # f(x - eps)
            b[i, j] = orig_val - eps
            out2 = test_function(a, b)
            loss2 = torch.sum(out2 * grad_output).item()

            # Restore original value
            b[i, j] = orig_val

            # Compute numerical gradient
            numerical_grad = (loss1 - loss2) / (2 * eps)
            analytical_grad = analytical_grad_b[i, j].item()

            # Check if gradients are close enough
            rel_error = abs(numerical_grad - analytical_grad) / max(
                1e-8, abs(analytical_grad)
            )
            self.assertLess(
                rel_error,
                0.1,
                f"Gradient check failed for b[{i},{j}]: numerical={numerical_grad}, analytical={analytical_grad}",
            )

        logging.info("Numerical gradient check passed for sampled elements")


if __name__ == "__main__":
    unittest.main()
