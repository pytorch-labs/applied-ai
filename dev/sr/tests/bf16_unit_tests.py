import torch
import unittest
import stochastic_rounding_cuda
from common_test_utils import StochasticRoundingTestBase

class TestBF16StochasticRounding(StochasticRoundingTestBase):
    """Test suite for BFloat16 stochastic rounding."""

    def setUp(self):
        super().setUp()
        self.round_function = stochastic_rounding_cuda.stochastic_round_bf16
        self.output_dtype = torch.bfloat16
        torch.cuda.manual_seed(2020)

    def test_rounding_statistics_1(self):
        """Test if rounding probabilities match expected distribution - case 1."""
        self.test_rounding_statistics_helper(2.1999969482421875, 2.1875, 2.2031)

    def test_rounding_statistics_2(self):
        """Test stochastic rounding with different BF16 boundary values."""
        self.test_rounding_statistics_helper(1.7999992370605469, 1.7969, 1.8047)

    def test_rounding_statistics_small(self):
        """Test stochastic rounding for number between 0 and 1."""
        self.test_rounding_statistics_helper(0.7499847412109375, 0.7480, 0.7500)

    def test_rounding_statistics_large(self):
        """Test stochastic rounding for large number, over 100."""
        self.test_rounding_statistics_helper(128.99998474121094, 128.875, 129.000)

    def test_bf16_specific_behavior(self):
        """Test specific behaviors of BF16 format."""
        # BF16 has 7 bits of precision in the mantissa (vs 10 in FP16)
        # Test a specific case that highlights this difference
        num_samples = 10000

        # Create tensors with values that should round differently in BF16 vs FP16
        # BF16 has fewer mantissa bits, so larger gaps between representable values
        x = torch.tensor([1.0 + 2**-8], device=self.device)  # This bit would be preserved in FP16 but not BF16

        rounded = self.round_function(x)

        # Should round to either 1.0 or the next BF16 representable value
        self.assertIn(rounded.item(), [1.0, 1.0 + 2**-7])


if __name__ == '__main__':

    unittest.main(verbosity=2)
