import unittest
import torch
from stochastic_rounding_cuda import stochastic_round_fp16

class TestFP16StochasticRounding(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda')
        torch.cuda.manual_seed(42)
    def _test_rounding_statistics_helper(self, value, lower_value, upper_value, tensor_size=10000, rounds=100, max_variance=0.03):
        """Helper method for testing FP16 stochastic rounding statistics"""
        print(f"\nInput value: {value}")
        MAX_VARIANCE = max_variance
        x = torch.full((tensor_size,), value, device='cuda')
        torch.cuda.manual_seed(42)

        # Single round test
        single_result = stochastic_round_fp16(x)
        print(f"Possible rounded values: {torch.unique(single_result)}")

        # Multiple rounds
        results = torch.empty((rounds, tensor_size), device='cuda', dtype=torch.float16)
        for i in range(rounds):
            results[i] = stochastic_round_fp16(x)

        prob_up = (results == upper_value).float().mean().item()
        print(f"Kernel's probability of rounding up: {prob_up:.4f}")

        distance_to_lower = abs(value - lower_value)
        total_distance = upper_value - lower_value
        expected_prob = distance_to_lower / total_distance
        print(f"Expected probability: {expected_prob:.4f}")

        self.assertTrue(abs(prob_up - expected_prob) < MAX_VARIANCE)

    def test_fp16_rounding_statistics(self):
        """Test FP16 rounding between 1 and 2"""
        self._test_rounding_statistics_helper(1.5000152587890625, 1.5, 1.5009765625)

    def test_fp16_rounding_statistics_small(self):
        """Test FP16 rounding for small values"""
        self._test_rounding_statistics_helper(0.2500152587890625, 0.25, 0.250244140625)


    def test_fp16_rounding_statistics_large(self):
        """Test FP16 rounding for large values"""
        self._test_rounding_statistics_helper(128.500152587890625, 128.5, 128.5625, max_variance=0.05)
    def test_rounding_distribution(self):
        # Test value between two representable FP16 values
        # 1.0 is exactly representable in FP16
        # Next representable value is 1.0 + 2^-10 ≈ 1.000977
        num_samples = 100000
        input_value = 1.0 + 2**-11  # Halfway between 1.0 and next FP16 value

        x = torch.full((num_samples,), input_value, dtype=torch.float32, device=self.device)
        rounded = stochastic_round_fp16(x, requires_grad=False)

        # Count occurrences of each value
        unique = torch.unique(rounded, sorted=True)
        self.assertEqual(len(unique), 2, "Should round to exactly two values")

        # Check distribution
        expected_low = 1.0
        expected_high = 1.0 + 2**-10

        counts = torch.zeros(2, device=self.device)
        counts[0] = (rounded == expected_low).sum()
        counts[1] = (rounded == expected_high).sum()

        # Allow for 1% tolerance in the distribution
        tolerance = 0.01
        ratio_low = counts[0].item() / num_samples
        ratio_high = counts[1].item() / num_samples

        self.assertLess(abs(ratio_low - 0.5), tolerance)
        self.assertLess(abs(ratio_high - 0.5), tolerance)

    def test_rounding_extremes(self):
        # Test values at different orders of magnitude
        test_cases = [
            (1e-3, 1000),    # Small values
            (1.0, 1000),     # Medium values
            (1e3, 1000),     # Large values
        ]

        for base_val, num_samples in test_cases:
            fp16_step = base_val * 2**-10
            input_value = base_val + fp16_step/2

            x = torch.full((num_samples,), input_value, dtype=torch.float32, device=self.device)
            rounded = stochastic_round_fp16(x, requires_grad=False)

            unique = torch.unique(rounded, sorted=True)
            counts = torch.zeros(len(unique), device=self.device)
            for i, val in enumerate(unique):
                counts[i] = (rounded == val).sum()

            # Check if distribution is roughly uniform
            expected = float(num_samples) / len(unique)
            max_diff = torch.max(torch.abs(counts - expected))
            self.assertLess(max_diff.item(), expected * 0.1,
                          f"Failed at value {base_val}")

    def test_bias(self):
        """Test that the rounding is unbiased"""
        num_samples = 100000
        test_values = [1.0, 2.0, 4.0, 8.0]

        for value in test_values:
            fp16_step = value * 2**-10
            input_value = value + fp16_step/2

            x = torch.full((num_samples,), input_value, dtype=torch.float32, device=self.device)
            rounded = stochastic_round_fp16(x, requires_grad=False)

            # Check mean is close to input value
            mean = rounded.mean()
            self.assertLess(abs(mean.item() - input_value), fp16_step * 0.01)

    def test_mantissa_bits(self):
        """Test that mantissa bits are properly handled"""
        num_samples = 10000

        # Create value that exercises all mantissa bits
        base = 1.0
        for i in range(10):  # FP16 has 10 mantissa bits
            input_value = base + 2**(-11-i)  # Just past midpoint
            x = torch.full((num_samples,), input_value, dtype=torch.float32, device=self.device)
            rounded = stochastic_round_fp16(x, requires_grad=False)

            unique = torch.unique(rounded, sorted=True)
            self.assertEqual(len(unique), 2, f"Failed at bit position {i}")

            low_count = (rounded == unique[0]).sum()
            ratio = low_count.item() / num_samples

            self.assertGreaterEqual(ratio, 0.45, f"Bad distribution at bit {i}")
            self.assertLessEqual(ratio, 0.55, f"Bad distribution at bit {i}")

    def test_subnormal_handling(self):
        """Test handling of values near subnormal boundary"""
        num_samples = 10000

        # Smallest normal FP16 value is 2^-14
        min_normal = 2**-14
        input_value = min_normal * 1.5  # Between smallest normal and next value

        x = torch.full((num_samples,), input_value, dtype=torch.float32, device=self.device)
        rounded = stochastic_round_fp16(x, requires_grad=False)

        # Should still get valid numbers without underflow
        self.assertFalse(torch.isnan(rounded).any())
        self.assertFalse(torch.isinf(rounded).any())
        self.assertTrue((rounded > 0).all())

    def test_reproducibility(self):
        """Test that results are reproducible with same seed"""
        num_samples = 1000
        input_value = 1.0 + 2**-11

        torch.cuda.manual_seed(42)
        x = torch.full((num_samples,), input_value, dtype=torch.float32, device=self.device)
        result1 = stochastic_round_fp16(x, requires_grad=False)

        torch.cuda.manual_seed(42)
        result2 = stochastic_round_fp16(x, requires_grad=False)

        self.assertTrue(torch.equal(result1, result2))

if __name__ == '__main__':
    unittest.main()
