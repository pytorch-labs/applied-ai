import torch
import numpy as np
from collections import Counter
import unittest
import stochastic_rounding_cuda
import time

class TestStochasticRounding(unittest.TestCase):
    def setup(self):
        # Ensure deterministic behavior for some tests
        torch.manual_seed(42)
        np.random.seed(42)

    def _test_rounding_statistics_helper(self, value, lower_value, upper_value, tensor_size=10000, rounds=100):
        """Helper method for testing stochastic rounding statistics"""
        print(f"\nInput value: {value}")
        MAX_VARIANCE = 0.03
        x = torch.full((tensor_size,), value, device='cuda')
        torch.cuda.manual_seed(42)

        # Single round test - isolate and show the round up and round down values
        single_result = stochastic_rounding_cuda.stochastic_round_bf16(x)
        print(f"Possible rounded values: {torch.unique(single_result)}")

        # Multiple rounds
        results = torch.empty((rounds, tensor_size), device='cuda', dtype=torch.bfloat16)
        for i in range(rounds):
            results[i] = stochastic_rounding_cuda.stochastic_round_bf16(x)

        prob_up = (results == upper_value).float().mean().item()
        print(f"Kernel's probability of rounding up: {prob_up:.4f}")

        distance_to_lower = abs(value - lower_value)
        total_distance = upper_value - lower_value
        expected_prob = distance_to_lower / total_distance
        print(f"Expected probability: {expected_prob:.4f}")

        self.assertTrue(abs(prob_up - expected_prob) < MAX_VARIANCE)

    def test_special_values(self):
        """Test handling of special values like inf, -inf, nan"""
        special_values = torch.tensor([float('inf'), float('-inf'), float('nan'), 0.0, -0.0],
                                    device='cuda')
        rounded = stochastic_rounding_cuda.stochastic_round_bf16(special_values)

        # Check inf and -inf are preserved
        self.assertTrue(torch.isinf(rounded[0]))
        self.assertTrue(torch.isinf(rounded[1]))
        self.assertTrue(rounded[0] > 0)
        self.assertTrue(rounded[1] < 0)

        # Check nan is preserved
        self.assertTrue(torch.isnan(rounded[2]))

        # Check zeros are preserved
        self.assertEqual(rounded[3].item(), 0.0)
        self.assertEqual(rounded[4].item(), 0.0)

    def test_small_values(self):
        """Test handling of small values near zero"""
        small_values = torch.tensor([1e-38, -1e-38, 1e-20, -1e-20], device='cuda')
        rounded = stochastic_rounding_cuda.stochastic_round_bf16(small_values)

        # Check that very small values are handled properly
        self.assertTrue(torch.all(torch.isfinite(rounded)))

    def test_vectorized_loading(self):
        """Test if vectorized loading works correctly for different tensor sizes"""
        sizes = [4, 8, 9, 16, 32, 100]  # Test various sizes including non-aligned

        for size in sizes:
            x = torch.linspace(1, size, size, device='cuda')
            rounded = stochastic_rounding_cuda.stochastic_round_bf16(x)

            # Check output size matches input
            self.assertEqual(rounded.size(0), size)

            # Check dtype
            self.assertEqual(rounded.dtype, torch.bfloat16)

    def test_large_values(self):
        """Test handling of large values"""
        large_values = torch.tensor([1e38, -1e38, 1e20, -1e20], device='cuda')
        rounded = stochastic_rounding_cuda.stochastic_round_bf16(large_values)

        # Values should be preserved approximately in BF16 range
        self.assertTrue(torch.all(torch.isfinite(rounded)))

    def test_rounding_statistics(self):
        """Test if rounding probabilities match expected distribution"""
        self._test_rounding_statistics_helper(2.1999969482421875, 2.1875, 2.2031)

    def test_rounding_statistics_2(self):
        """Test stochastic rounding with different BF16 boundary values"""
        self._test_rounding_statistics_helper(1.7999992370605469, 1.7969, 1.8047)

    def test_rounding_statistics_small(self):
        """Test stochastic rounding for number between 0 and 1"""
        self._test_rounding_statistics_helper(0.7499847412109375, 0.7480, 0.7500)

    def test_rounding_statistics_large(self):
        """Test stochastic rounding for large number, over 100"""
        self._test_rounding_statistics_helper(128.99998474121094, 128.875, 129.000)



if __name__ == '__main__':
    unittest.main(verbosity=2)
