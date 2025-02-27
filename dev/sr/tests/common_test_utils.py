import torch
import numpy as np
import unittest

class StochasticRoundingTestBase(unittest.TestCase):
    """Base class for testing stochastic rounding implementations."""


    def setUp(self):
        """Common setup for all stochastic rounding tests."""
        self.device = torch.device('cuda')
        torch.cuda.manual_seed(2020)
        np.random.seed(2020)

        # To be defined by child classes
        self.round_function = None
        self.output_dtype = None

    def test_rounding_statistics_helper(self, value, lower_value, upper_value, tensor_size=10000, rounds=100, max_variance=0.03):
        """Helper method for testing stochastic rounding statistics.

        Args:
            value: The input value to round (float)
            lower_value: The expected lower bound of rounding
            upper_value: The expected upper bound of rounding
            tensor_size: Size of the test tensor
            rounds: Number of rounds to perform for statistics
            max_variance: Maximum acceptable variance from expected probability
        """
        print(f"\nInput value: {value}")
        x = torch.full((tensor_size,), value, device=self.device)
        torch.cuda.manual_seed(42)

        # Single round test to determine possible rounded values
        single_result = self.round_function(x)
        print(f"Possible rounded values: {torch.unique(single_result)}")

        # Multiple rounds to gather statistics
        results = torch.empty((rounds, tensor_size), device=self.device, dtype=self.output_dtype)
        for i in range(rounds):
            results[i] = self.round_function(x)

        prob_up = (results == upper_value).float().mean().item()
        print(f"Kernel's probability of rounding up: {prob_up:.4f}")

        distance_to_lower = abs(value - lower_value)
        total_distance = upper_value - lower_value
        expected_prob = distance_to_lower / total_distance
        print(f"Expected probability: {expected_prob:.4f}")

        self.assertTrue(abs(prob_up - expected_prob) < max_variance)

    def test_special_values(self):
        """Test handling of special values like inf, -inf, nan."""
        special_values = torch.tensor([float('inf'), float('-inf'), float('nan'), 0.0, -0.0],
                                      device=self.device)
        rounded = self.round_function(special_values)

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
        """Test handling of small values near zero."""
        small_values = torch.tensor([1e-38, -1e-38, 1e-20, -1e-20], device=self.device)
        rounded = self.round_function(small_values)

        # Check that very small values are handled properly
        self.assertTrue(torch.all(torch.isfinite(rounded)))

    def test_large_values(self):
        """Test handling of large values."""
        large_values = torch.tensor([1e38, -1e38, 1e20, -1e20], device=self.device)
        rounded = self.round_function(large_values)

        # Values should be preserved approximately in the target dtype range
        self.assertTrue(torch.all(torch.isfinite(rounded)))

    def test_vectorized_loading(self):
        """Test if vectorized loading works correctly for different tensor sizes."""
        sizes = [4, 8, 9, 16, 32, 100]  # Test various sizes including non-aligned

        for size in sizes:
            x = torch.linspace(1, size, size, device=self.device)
            rounded = self.round_function(x)

            # Check output size matches input
            self.assertEqual(rounded.size(0), size)

            # Check dtype
            self.assertEqual(rounded.dtype, self.output_dtype)

    def test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        num_samples = 1000
        input_value = 1.5000152587890625  # A value between representable points

        torch.cuda.manual_seed(42)
        x = torch.full((num_samples,), input_value, dtype=torch.float32, device=self.device)
        result1 = self.round_function(x)

        torch.cuda.manual_seed(42)
        result2 = self.round_function(x)

        self.assertTrue(torch.equal(result1, result2))

    def test_bias(self):
        """Test that the rounding is unbiased over many samples."""
        num_samples = 100000
        test_values = [1.0, 2.0, 4.0, 8.0]

        for base_value in test_values:
            # Find the next representable value (this is an approximation)
            if self.output_dtype == torch.bfloat16:
                step = base_value * 2**-7
            else:  # float16
                step = base_value * 2**-10

            input_value = base_value + step/2

            x = torch.full((num_samples,), input_value, dtype=torch.float32, device=self.device)
            rounded = self.round_function(x)

            # The mean should be close to the input value
            mean = rounded.float().mean().item()
            self.assertLess(abs(mean - input_value), step * 0.01)
