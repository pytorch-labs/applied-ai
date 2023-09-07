# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, NamedTuple, Optional, Tuple, Union

import pytest
import torch
from alibi_positional_embeddings import AlibiPositionEmbeddings
from torch import nn


def assert_expected(
    actual: Any,
    expected: Any,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    check_device=True,
):
    torch.testing.assert_close(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
        check_device=check_device,
        msg=f"actual: {actual}, expected: {expected}",
    )


@pytest.fixture(autouse=True)
def random():
    torch.manual_seed(2023)


class TestAlibiPositionEmbedding:
    @pytest.fixture
    def max_seq_len(self):
        return 16

    @pytest.fixture
    def embedding_dim(self):
        return 32

    @pytest.fixture
    def num_heads(self):
        return 8

    def test_alibi_mask(
        self,
        max_seq_len,
        num_heads,
    ):
        alibi_class = AlibiPositionEmbeddings(
            max_seq_len=max_seq_len, num_heads=num_heads
        )
        base_mask = alibi_class.get_attention_mask(max_seq_len)

        # verify mask shape
        expected_shape = torch.Size((num_heads, max_seq_len, max_seq_len))
        assert_expected(base_mask.shape, expected_shape)

        # verify alibi mask components
        expected_last_head_row = torch.tensor(
            [
                -0.0586,
                -0.0547,
                -0.0508,
                -0.0469,
                -0.0430,
                -0.0391,
                -0.0352,
                -0.0312,
                -0.0273,
                -0.0234,
                -0.0195,
                -0.0156,
                -0.0117,
                -0.0078,
                -0.0039,
                0.0000,
            ]
        )

        expected_first_head_first_row_first_entry = torch.tensor(
            0.0000,
        )

        assert_expected(
            base_mask[0][0][0],
            expected_first_head_first_row_first_entry,
            rtol=0,
            atol=1e-4,
        )

        assert_expected(
            base_mask[num_heads - 1][max_seq_len - 1],
            expected_last_head_row,
            rtol=0,
            atol=1e-4,
        )
