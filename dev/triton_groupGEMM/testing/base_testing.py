# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
# This code is derived from: https://github.com/pytorch/FBGEMM/tree/main/fbgemm_gpu/experimental/gemm/triton_gemm


import logging

# support parent dir access (TODO - make package)
import os
import sys
import unittest
from typing import Tuple

import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groupgemm import grouped_gemm


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


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
