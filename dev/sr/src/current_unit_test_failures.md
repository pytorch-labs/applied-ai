python ./tests/fp16_unit_tests.py
F
Input value: 1.5000152587890625
Possible rounded values: tensor([1.5000, 1.5010], device='cuda:0', dtype=torch.float16)
Kernel's probability of rounding up: 0.0158
Expected probability: 0.0156
.
Input value: 128.50015258789062
Possible rounded values: tensor([128.5000, 128.6250], device='cuda:0', dtype=torch.float16)
Kernel's probability of rounding up: 0.9990
Expected probability: 0.0024
F
Input value: 0.2500152587890625
Possible rounded values: tensor([0.2500, 0.2502], device='cuda:0', dtype=torch.float16)
Kernel's probability of rounding up: 0.0625
Expected probability: 0.0625
.FF.F.
======================================================================
FAIL: test_bias (__main__.TestFP16StochasticRounding.test_bias)
Test that the rounding is unbiased
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/data/users/less/applied-ai/dev/sr/./tests/fp16_unit_tests.py", line 117, in test_bias
    self.assertLess(abs(mean.item() - input_value), fp16_step * 0.01)
AssertionError: 0.00048828125 not less than 9.765625e-06

======================================================================
FAIL: test_fp16_rounding_statistics_large (__main__.TestFP16StochasticRounding.test_fp16_rounding_statistics_large)
Test FP16 rounding for large values
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/data/users/less/applied-ai/dev/sr/./tests/fp16_unit_tests.py", line 46, in test_fp16_rounding_statistics_large
    self._test_rounding_statistics_helper(128.500152587890625, 128.5, 128.5625, max_variance=0.05)
  File "/data/users/less/applied-ai/dev/sr/./tests/fp16_unit_tests.py", line 33, in _test_rounding_statistics_helper
    self.assertTrue(abs(prob_up - expected_prob) < MAX_VARIANCE)
AssertionError: False is not true

======================================================================
FAIL: test_mantissa_bits (__main__.TestFP16StochasticRounding.test_mantissa_bits)
Test that mantissa bits are properly handled
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/data/users/less/applied-ai/dev/sr/./tests/fp16_unit_tests.py", line 137, in test_mantissa_bits
    self.assertLessEqual(ratio, 0.55, f"Bad distribution at bit {i}")
AssertionError: 0.7476 not less than or equal to 0.55 : Bad distribution at bit 1

======================================================================
FAIL: test_reproducibility (__main__.TestFP16StochasticRounding.test_reproducibility)
Test that results are reproducible with same seed
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/data/users/less/applied-ai/dev/sr/./tests/fp16_unit_tests.py", line 167, in test_reproducibility
    self.assertTrue(torch.equal(result1, result2))
AssertionError: False is not true

======================================================================
FAIL: test_rounding_extremes (__main__.TestFP16StochasticRounding.test_rounding_extremes)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/data/users/less/applied-ai/dev/sr/./tests/fp16_unit_tests.py", line 100, in test_rounding_extremes
    self.assertLess(max_diff.item(), expected * 0.1,
AssertionError: 408.0 not less than 50.0 : Failed at value 0.001

----------------------------------------------------------------------
Ran 9 tests in 1.330s

FAILED (failures=5)
