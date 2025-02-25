import unittest
import torch
from stochastic_rounding_cuda import stochastic_round_fp16


def debug_fp16_rounding():
    x = torch.tensor([1.5000152587890625], device='cuda')
    result = stochastic_round_fp16(x)
    print(f"Input: {x.item()}, Output: {result.item()}")

if __name__ == '__main__':
    debug_fp16_rounding()
