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
        # Add seed for reproducibility
        torch.cuda.manual_seed(42)

        value = 2.1999969482421875
        tensor_size = 10000 # TODO - should be 10K
        x = torch.full((tensor_size,), value, device='cuda')

        # Debug prints
        print(f"\nInput value: {value}")

        # Single round test first
        single_result = stochastic_rounding_cuda.stochastic_round_bf16(x)
        unique_vals = torch.unique(single_result)
        print(f"Possible rounded values: {unique_vals}")

        # Multiple rounds
        rounds = 100
        results = torch.empty((rounds, tensor_size), device='cuda', dtype = torch.bfloat16)
        for i in range(rounds):
            results[i] = stochastic_rounding_cuda.stochastic_round_bf16(x)

        lower_value = 2.1875
        upper_value = 2.2031

        prob_up = (results == upper_value).float().mean().item()
        print(f"Kernel's probability of rounding up: {prob_up:.4f}")

        # Calculate expected probability based on input position
        distance_to_lower = abs(value - lower_value)
        total_distance = upper_value - lower_value
        expected_prob = distance_to_lower / total_distance
        print(f"Expected probability: {expected_prob:.4f}")

        self.assertTrue(abs(prob_up - expected_prob) < 0.03)


    def test_rounding_statistics_2(self):
        """Test stochastic rounding with different BF16 boundary values"""
        # Add seed for reproducibility
        torch.cuda.manual_seed(42)
        value = 1.7999992370605469
        # debug if needed:
        # print(f"Value bits: {torch.tensor(value).view(torch.int32).item():x}")
        tensor_size = 10000
        x = torch.full((tensor_size,), value, device='cuda')

        # Debug prints
        print(f"\nInput value: {value}")

        # Single round test first
        single_result = stochastic_rounding_cuda.stochastic_round_bf16(x)
        unique_vals = torch.unique(single_result)
        print(f"Possible rounded values: {unique_vals}")

        rounds = 100
        results = torch.empty((rounds, tensor_size), device='cuda', dtype=torch.bfloat16)
        for i in range(rounds):
            results[i] = stochastic_rounding_cuda.stochastic_round_bf16(x)

        lower_value = 1.7969
        upper_value = 1.8047

        prob_up = (results == upper_value).float().mean().item()
        print(f"Kernel's probability of rounding up: {prob_up:.4f}")

        distance_to_lower = abs(value - lower_value)
        total_distance = upper_value - lower_value
        expected_prob = distance_to_lower / total_distance
        print(f"Expected probability: {expected_prob:.4f}")

        self.assertTrue(abs(prob_up - expected_prob) < 0.03)

    def test_rounding_statistics_small(self):
        """Test stochastic rounding for number between 0 and 1"""
        value = 0.7499847412109375  # Should round between 0.7480 and 0.7500

        print(f"\nInput value: {value}")
        tensor_size = 10000
        x = torch.full((tensor_size,), value, device='cuda')
        torch.cuda.manual_seed(42)

        rounds = 100
        results = torch.empty((rounds, tensor_size), device='cuda', dtype=torch.bfloat16)
        for i in range(rounds):
            results[i] = stochastic_rounding_cuda.stochastic_round_bf16(x)

        lower_value = 0.7480
        upper_value = 0.7500
        prob_up = (results == upper_value).float().mean().item()
        print(f"Kernel's probability of rounding up: {prob_up:.4f}")

        distance_to_lower = abs(value - lower_value)
        total_distance = upper_value - lower_value
        expected_prob = distance_to_lower / total_distance
        print(f"Expected probability: {expected_prob:.4f}")

        self.assertTrue(abs(prob_up - expected_prob) < 0.03)

    def test_rounding_statistics_large(self):
        """Test stochastic rounding for large number, over 100"""
        value = 128.99998474121094  # Should round between 128.875 and 129.000
        # Debug prints
        print(f"\nInput value: {value}")
        tensor_size = 10000
        x = torch.full((tensor_size,), value, device='cuda')
        torch.cuda.manual_seed(42)

        rounds = 100
        results = torch.empty((rounds, tensor_size), device='cuda', dtype=torch.bfloat16)
        for i in range(rounds):
            results[i] = stochastic_rounding_cuda.stochastic_round_bf16(x)

        lower_value = 128.875
        upper_value = 129.000
        prob_up = (results == upper_value).float().mean().item()
        print(f"Kernel's probability of rounding up: {prob_up:.4f}")


        distance_to_lower = abs(value - lower_value)
        total_distance = upper_value - lower_value
        expected_prob = distance_to_lower / total_distance
        print(f"Expected probability: {expected_prob:.4f}")

        self.assertTrue(abs(prob_up - expected_prob) < 0.03)



if __name__ == '__main__':
    unittest.main(verbosity=2)
