# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
# This code is derived from: https://github.com/pytorch/FBGEMM/tree/main/fbgemm_gpu/experimental/gemm/triton_gemm


import logging
import unittest
from typing import Tuple

import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groupgemm import grouped_gemm


class TestGroupedGEMM(unittest.TestCase):
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
            self.assertTrue(result.shape == (M, N))
            expected_result = torch.zeros(M, N, dtype=torch.bfloat16, device=device)
            for g in range(G):
                m_start = m_starts[g]
                m_end = m_ends[g]
                expected_result[m_start:m_end, :] = (
                    a[m_start:m_end, :] @ b[g * N : (g + 1) * N, :].T
                )
            torch.testing.assert_close(result, expected_result, atol=1e-5, rtol=1.6e-2)

        for G in (1, 4, 16):
            for M in (64, 512):
                logging.info(f"Testing BF16 GMM with G={G}, M={M}")
                _test_grouped_gemm_bf16((G, M, 256, 256), torch.device("cuda"))

    def test_grouped_gemm_bf16_various_dimensions(self) -> None:
        """Test grouped_gemm with bf16 precision and various dimensions"""

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
            self.assertTrue(result.shape == (M, N))
            expected_result = torch.zeros(M, N, dtype=torch.bfloat16, device=device)
            for g in range(G):
                m_start = m_starts[g]
                m_end = m_ends[g]
                expected_result[m_start:m_end, :] = (
                    a[m_start:m_end, :] @ b[g * N : (g + 1) * N, :].T
                )
            torch.testing.assert_close(result, expected_result, atol=1e-5, rtol=1.6e-2)

        for G in (4, 8):
            for M in (128, 256):
                for N, K in [(128, 256), (256, 128), (64, 64)]:
                    logging.info(f"Testing BF16 GMM with G={G}, M={M}, N={N}, K={K}")
                    _test_grouped_gemm_bf16((G, M, N, K), torch.device("cuda"))

    def test_grouped_gemm_bf16_edge_cases(self) -> None:
        """Test grouped_gemm with bfloat16 for various edge cases"""
        device = torch.device("cuda")

        # Test with G=1 (single group case)
        G, M, N, K = 1, 32, 32, 32
        a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        b = torch.randn(N * G, K, dtype=torch.bfloat16, device=device)
        m_sizes = torch.tensor([M], device=device).to(torch.int32)
        result = grouped_gemm(a, b, m_sizes)
        expected_result = a @ b.T
        torch.testing.assert_close(result, expected_result, atol=1e-5, rtol=1.6e-2)

        # Test with uneven group sizes
        G, M, N, K = 3, 100, 32, 32
        a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        b = torch.randn(N * G, K, dtype=torch.bfloat16, device=device)
        m_sizes = torch.tensor([25, 50, 25], device=device).to(torch.int32)
        result = grouped_gemm(a, b, m_sizes)
        self.assertTrue(result.shape == (M, N))
        expected_result = torch.zeros(M, N, dtype=torch.bfloat16, device=device)
        m_start = 0
        for g in range(G):
            m_end = m_start + m_sizes[g].item()
            expected_result[m_start:m_end, :] = (
                a[m_start:m_end, :] @ b[g * N : (g + 1) * N, :].T
            )
            m_start = m_end
        torch.testing.assert_close(result, expected_result, atol=1e-5, rtol=1.6e-2)

        # Test with extremely small matrices
        G, M, N, K = 2, 8, 8, 8
        a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        b = torch.randn(N * G, K, dtype=torch.bfloat16, device=device)
        m_sizes = torch.tensor([4, 4], device=device).to(torch.int32)
        result = grouped_gemm(a, b, m_sizes)
        expected_result = torch.zeros(M, N, dtype=torch.bfloat16, device=device)
        m_start = 0
        for g in range(G):
            m_end = m_start + m_sizes[g].item()
            expected_result[m_start:m_end, :] = (
                a[m_start:m_end, :] @ b[g * N : (g + 1) * N, :].T
            )
            m_start = m_end
        torch.testing.assert_close(result, expected_result, atol=1e-5, rtol=1.6e-2)

        # Test with large group count but small matrix sizes
        G, M, N, K = 32, 128, 16, 16
        a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        b = torch.randn(N * G, K, dtype=torch.bfloat16, device=device)
        m_sizes = torch.ones(G, device=device).to(torch.int32) * (M // G)
        m_sizes[-1] += M % G  # Adjust the last group size to account for remainder
        result = grouped_gemm(a, b, m_sizes)
        expected_result = torch.zeros(M, N, dtype=torch.bfloat16, device=device)
        m_start = 0
        for g in range(G):
            m_end = m_start + m_sizes[g].item()
            expected_result[m_start:m_end, :] = (
                a[m_start:m_end, :] @ b[g * N : (g + 1) * N, :].T
            )
            m_start = m_end
        torch.testing.assert_close(result, expected_result, atol=1e-5, rtol=1.6e-2)

    def test_grouped_gemm_bf16_invalid_inputs(self) -> None:
        """Test grouped_gemm with invalid inputs to ensure proper error handling"""
        device = torch.device("cuda")

        # Test with mismatched dimensions
        G, M, N, K = 2, 64, 32, 32
        a = torch.randn(
            M, K + 1, dtype=torch.bfloat16, device=device
        )  # Wrong K dimension
        b = torch.randn(N * G, K, dtype=torch.bfloat16, device=device)
        m_sizes = torch.tensor([32, 32], device=device).to(torch.int32)

        with self.assertRaises(RuntimeError):
            grouped_gemm(a, b, m_sizes)

        # Test with mismatched G and m_sizes length
        G, M, N, K = 2, 64, 32, 32
        a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        b = torch.randn(N * G, K, dtype=torch.bfloat16, device=device)
        m_sizes = torch.tensor([32, 32, 32], device=device).to(
            torch.int32
        )  # Too many groups

        with self.assertRaises((RuntimeError, ValueError, IndexError)):
            grouped_gemm(a, b, m_sizes)

        # Test with incorrect sum of m_sizes
        G, M, N, K = 2, 64, 32, 32
        a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        b = torch.randn(N * G, K, dtype=torch.bfloat16, device=device)
        m_sizes = torch.tensor([32, 40], device=device).to(torch.int32)  # Sum > M

        with self.assertRaises((RuntimeError, ValueError, IndexError)):
            grouped_gemm(a, b, m_sizes)

        # Test with negative m_sizes values
        G, M, N, K = 2, 64, 32, 32
        a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        b = torch.randn(N * G, K, dtype=torch.bfloat16, device=device)
        m_sizes = torch.tensor([40, -8], device=device).to(
            torch.int32
        )  # Negative group size

        with self.assertRaises((RuntimeError, ValueError)):
            grouped_gemm(a, b, m_sizes)

    def test_grouped_gemm_bf16_deterministic(self) -> None:
        """Test that grouped_gemm produces deterministic results with the same inputs"""
        G, M, N, K = 4, 128, 64, 64
        device = torch.device("cuda")

        # Fix the random seed for reproducibility
        torch.manual_seed(42)

        a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        b = torch.randn(N * G, K, dtype=torch.bfloat16, device=device)
        m_sizes = torch.tensor([32, 32, 32, 32], device=device).to(torch.int32)

        # First run
        result1 = grouped_gemm(a, b, m_sizes)

        # Second run with same inputs
        result2 = grouped_gemm(a, b, m_sizes)

        # Results should be exactly the same
        self.assertTrue(torch.all(result1 == result2))

    def test_grouped_gemm_bf16_large_matrices(self) -> None:
        """Test grouped_gemm with larger matrices to stress test performance and stability"""
        device = torch.device("cuda")

        # Test with large matrices but fewer groups
        G, M, N, K = 2, 2048, 512, 1024
        a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        b = torch.randn(N * G, K, dtype=torch.bfloat16, device=device)
        m_sizes = torch.tensor([1024, 1024], device=device).to(torch.int32)

        result = grouped_gemm(a, b, m_sizes)
        expected_result = torch.zeros(M, N, dtype=torch.bfloat16, device=device)

        m_start = 0
        for g in range(G):
            m_end = m_start + m_sizes[g].item()
            expected_result[m_start:m_end, :] = (
                a[m_start:m_end, :] @ b[g * N : (g + 1) * N, :].T
            )
            m_start = m_end

        torch.testing.assert_close(result, expected_result, atol=1e-5, rtol=1.6e-2)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
