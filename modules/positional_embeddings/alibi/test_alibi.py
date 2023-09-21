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

    @pytest.fixture
    def num_heads_non_power_2(self):
        return 12

    def test_alibi_mask_power_of_2(
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

    def test_alibi_mask_non_power_of_2(
        self,
        max_seq_len,
        num_heads_non_power_2,
    ):
        alibi_class = AlibiPositionEmbeddings(
            max_seq_len=max_seq_len, num_heads=num_heads_non_power_2
        )
        base_mask = alibi_class.get_attention_mask(max_seq_len)

        # verify mask shape
        expected_shape = torch.Size((num_heads_non_power_2, max_seq_len, max_seq_len))
        assert_expected(base_mask.shape, expected_shape)

        # verify alibi mask components
        expected_second_head_last_row = torch.tensor(
            [
                -7.5000,
                -7.0000,
                -6.5000,
                -6.0000,
                -5.5000,
                -5.0000,
                -4.5000,
                -4.0000,
                -3.5000,
                -3.0000,
                -2.5000,
                -2.0000,
                -1.5000,
                -1.0000,
                -0.5000,
                0.0000,
            ]
        )

        expected_third_head_last_row = torch.tensor(
            [
                -5.3033,
                -4.9497,
                -4.5962,
                -4.2426,
                -3.8891,
                -3.5355,
                -3.1820,
                -2.8284,
                -2.4749,
                -2.1213,
                -1.7678,
                -1.4142,
                -1.0607,
                -0.7071,
                -0.3536,
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

        # verify 2nd and 3rd head to confirm non power 2 symmetry of slopes
        assert_expected(
            base_mask[1][max_seq_len - 1],
            expected_second_head_last_row,
            rtol=0,
            atol=1e-4,
        )

        assert_expected(
            base_mask[2][max_seq_len - 1],
            expected_third_head_last_row,
            rtol=0,
            atol=1e-4,
        )
