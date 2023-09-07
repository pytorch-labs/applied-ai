# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import math
from typing import Tuple, List

import torch
from torch import nn, Tensor


class AlibiPositionEmbeddings(nn.Module):
    """Attention with Linear Biases (ALiBi)

    # Softmax(qiKT + m · [-(i - 1), ..., -2, -1, 0]),
    where m = fixed specific slope per head

    as proposed in:
    https://arxiv.org/abs/2108.12409
    Train Short, Test Long: Attention with Linear Biases
    Enables Input Length Extrapolation

    derived from Ofir Press (alibi author) codebase:
    https://github.com/ofirpress/attention_with_linear_biases

    """

    def __init__(
        self,
        max_seq_len: int,
        num_heads: int,
    ) -> None:
        """recommended usage:  create alibi mask before transformer block loop and integrate
        Alibi should be applied after the sqrt scaling of the attention values

        Example:
        before Transformer block loop:
            from alibi_embeddings import AlibiPE
            self.alibi = AlibiPE(config.max_seq_len, config.num_heads)
        pass a reference to the alibi class to each transformer layer
        then in forward of transformer layer:
            alibi_mask = self.alibi.get_attention_mask(N) # N = seq length of this batch
            ...
            attn = q @ k.transpose( -2, -1)
            att *= 1.0 / math.sqrt(k.size(-1))
            att += alibi_mask

        """
        super().__init__()

        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        self.causal_mask = self.build_causal_attention_mask(
            self.max_seq_len, self.num_heads
        )
        self.alibi_mask_base = self.build_alibi_mask(self.max_seq_len, self.num_heads)
        self.decoder_mask = self.causal_mask + self.alibi_mask_base
        self.register_buffer("alibi_mask", self.decoder_mask, persistent=False)

    def get_attention_mask(self, curr_seq_len: int) -> torch.Tensor:
        """returns the alibi mask, clipped to the current batch seq len"""
        return self.alibi_mask[..., :curr_seq_len, :curr_seq_len]

    @classmethod
    def build_causal_attention_mask(cls, seq_len: int, num_heads: int) -> torch.Tensor:
        """builds a generic causal attention mask"""
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1
        )
        attn_mask = causal_mask.repeat(num_heads, 1, 1)
        return attn_mask

    @classmethod
    def build_alibi_mask(cls, seq_len: int, num_heads: int) -> torch.Tensor:
        """generate the alibi mask by computing a distance bias matrix multiplied by each head's m (slope)"""
        distance_bias_matrix = -torch.abs(
            torch.arange(seq_len) - torch.arange(seq_len).view(-1, 1)
        )
        slope_per_head = Tensor(cls.get_slopes(num_heads)).view(-1, 1, 1)
        alibi_mask = distance_bias_matrix * slope_per_head
        return alibi_mask

    @staticmethod
    def get_slopes(num_heads: int) -> List[float]:
        """for n heads, a range from (0,1) and is the geometric sequence
        that starts at 2^(-8/n) and uses this same value as its ratio

        example: num_heads =4
        result: [0.25, 0.0625, 0.015625, 0.00390625]

        """

        def get_slopes_power_of_2(n: int) -> List[float]:
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(num_heads).is_integer():
            return get_slopes_power_of_2(num_heads)

        # paper authors note that they only trained models that have 2^a heads for some a.
        # This has beneficial properties related to input being power of 2.
        # Closest power of 2 below is workaround for when num of heads is not power of 2

        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][
                : num_heads - closest_power_of_2
            ]
        )
