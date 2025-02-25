import unittest
import torch
from stochastic_rounding_cuda import stochastic_round_fp16

class TestFP16StochasticRoundingImproved(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda')
        torch.cuda.manual_seed(42)

    def _test_rounding_statistics_helper(self, value, lower_value, upper_value, tensor_size=10000, rounds=100,
                                       max_variance=0.03, print_details=True):
        """Helper method for testing FP16 stochastic rounding statistics with detailed output"""
        if print_details:
            print(f"\nInput value: {value}")
            print(f"Lower FP16 value: {lower_value}")
            print(f"Upper FP16 value: {upper_value}")

        x = torch.full((tensor_size,), value, device='cuda')
        torch.cuda.manual_seed(42)

        # Single round test to check possible outcomes
        single_result = stochastic_round_fp16(x)
        if print_details:
            print(f"Possible rounded values: {torch.unique(single_result)}")

        # Multiple rounds to verify distribution
        results = torch.empty((rounds, tensor_size), device='cuda', dtype=torch.float16)
        for i in range(rounds):
            results[i] = stochastic_round_fp16(x)

        # Calculate actual probability of rounding up
        prob_up = (results == upper_value).float().mean().item()
        if print_details:
            print(f"Kernel's probability of rounding up: {prob_up:.4f}")

        # Calculate expected probability
        distance_to_lower = value - lower_value
        total_distance = upper_value - lower_value
        expected_prob = distance_to_lower / total_distance
        if print_details:
            print(f"Expected probability: {expected_prob:.4f}")
            print(f"Variance: {abs(prob_up - expected_prob):.4f} (max allowed: {max_variance})")

        # Verify that actual probability is close to expected
        self.assertLess(abs(prob_up - expected_prob), max_variance,
                      f"Probability {prob_up:.4f} differs from expected {expected_prob:.4f} by more than {max_variance}")

        return prob_up, expected_prob

    # High probability test cases - essentially, make it easier to verify the distribution
    def test_high_probability_75_percent(self):
        """Test with value 75% of the way from lower to upper representable value"""
        lower = 1.0
        upper = 1.0 + 2**-10  # Next representable FP16 after 1.0
        value = lower + 0.75 * (upper - lower)
        self._test_rounding_statistics_helper(value, lower, upper)

    def test_high_probability_90_percent(self):
        """Test with value 90% of the way from lower to upper representable value"""
        lower = 1.0
        upper = 1.0 + 2**-10  # Next representable FP16 after 1.0
        value = lower + 0.9 * (upper - lower)
        self._test_rounding_statistics_helper(value, lower, upper)

    def test_high_probability_99_percent(self):
        """Test with value 99% of the way from lower to upper representable value"""
        lower = 2.0
        upper = 2.0 + 2**-9  # Next representable FP16 after 2.0
        value = lower + 0.99 * (upper - lower)
        self._test_rounding_statistics_helper(value, lower, upper)

    # Test exact half probability (50%) cases
    def test_exact_half_probability(self):
        """Test with values exactly halfway between representable FP16 values"""
        test_bases = [1.0, 2.0, 4.0, 8.0]

        for base in test_bases:
            # Find the next representable FP16 value
            step = base * 2**-10  # The step size depends on the magnitude
            lower = base
            upper = base + step
            value = lower + step/2  # Exactly halfway

            print(f"\n--- Testing exact half probability at base {base} ---")
            prob_up, expected_prob = self._test_rounding_statistics_helper(
                value, lower, upper, tensor_size=20000, rounds=200, max_variance=0.02)

            # For exact half probability, we should be very close to 0.5
            self.assertLess(abs(prob_up - 0.5), 0.02,
                          f"Half probability test at {base} gave {prob_up:.4f}, expected close to 0.5")

    # Variable probability test
    def test_variable_probabilities(self):
        """Test with various probability levels between 0% and 100%"""
        base = 1.0
        step = base * 2**-10
        lower = base
        upper = base + step

        probabilities = [0.1, 0.25, 0.5, 0.75, 0.9]

        print("\n--- Variable probability tests ---")
        for expected_prob in probabilities:
            value = lower + expected_prob * step
            actual_prob, _ = self._test_rounding_statistics_helper(
                value, lower, upper, tensor_size=50000, print_details=False)

            print(f"Target: {expected_prob:.2f}, Actual: {actual_prob:.4f}, "
                  f"Diff: {abs(actual_prob - expected_prob):.4f}")

            # Test with tighter tolerance for these controlled tests
            self.assertLess(abs(actual_prob - expected_prob), 0.02,
                          f"Variable probability test at {expected_prob} failed")

    # Test for mantissa bits at different positions
    def test_specific_mantissa_bits(self):
        """Test handling of specific mantissa bit patterns"""
        # Test values that exercise specific bit patterns in the mantissa
        print("\n--- Specific mantissa bit pattern tests ---")

        # Start with 1.0 and set specific bits in the truncated part of mantissa
        base = 1.0
        step = base * 2**-10  # Distance to next representable FP16

        # Test with only the MSB of truncated bits set (approx 50% probability)
        msb_value = base + step * 0.5  # Sets bit 12 (MSB of truncated bits)
        self._test_rounding_statistics_helper(msb_value, base, base + step,
                                           tensor_size=20000, max_variance=0.02)

        # Test with only the LSB of truncated bits set (very low probability)
        lsb_value = base + step * (2**-12)  # Sets bit 0 (LSB of truncated bits)
        # Use higher variance for this very low probability case
        self._test_rounding_statistics_helper(lsb_value, base, base + step,
                                           tensor_size=50000, max_variance=0.1)

        # Test with alternating bit pattern (010101...) - approx 1/3 probability
        pattern_value = base + step * (1/3)  # Approximates an alternating bit pattern
        self._test_rounding_statistics_helper(pattern_value, base, base + step,
                                           tensor_size=30000, max_variance=0.03)

    def test_mantissa_boundary_values(self):
        """Test values just above/below boundaries in the mantissa"""
        print("\n--- Testing mantissa boundary values ---")

        # Just above 1.0 (smallest positive increment in truncated bits)
        epsilon = 2**-23  # Smallest representable difference in FP32
        just_above_1 = 1.0 + epsilon
        self._test_rounding_statistics_helper(just_above_1, 1.0, 1.0 + 2**-10,
                                           tensor_size=100000, max_variance=0.05)

        # Just below the next representable FP16 value
        almost_next = 1.0 + 2**-10 - epsilon
        self._test_rounding_statistics_helper(almost_next, 1.0, 1.0 + 2**-10,
                                           tensor_size=100000, max_variance=0.05)

        # Right in the middle
        half_point = 1.0 + 2**-11
        self._test_rounding_statistics_helper(half_point, 1.0, 1.0 + 2**-10,
                                           tensor_size=50000, max_variance=0.02)

if __name__ == '__main__':
    unittest.main()
