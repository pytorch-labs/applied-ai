Branch for stochastic rounding kernel
Currently processes 4 elements per thread to leverage rand4


Current Speed:

Running on: NVIDIA H100
Compute Capability: 9.0
Size              SR Time (ms)      SR ME/s    Cast Time (ms)    Cast ME/s    SR faster by
--------------  --------------  -----------  ----------------  -----------  --------------
1,000                    1.377        0.726             0.005      216.654           0.003
10,000                   1.373        7.284             0.005     2129.716           0.003
100,000                  1.366       73.222             0.005    21526.209           0.003
1,000,000                1.373      728.306             0.005   214933.207           0.003
10,000,000               1.531     6530.253             0.032   314742.243           0.021
70,000,000               1.957    35762.597             0.189   370399.199           0.097
100,000,000              1.901    52591.593             0.268   373377.887           0.141
1,000,000,000            4.971   201147.084             2.647   377797.563           0.532
10,000,000,000           6.322  1581857.692            26.609   375812.352           4.209

Unit Tests:
test_large_values (__main__.TestStochasticRounding.test_large_values)
Test handling of large values ... ok
test_rounding_statistics (__main__.TestStochasticRounding.test_rounding_statistics)
Test if rounding probabilities match expected distribution ...
Input value: 2.1999969482421875
Possible rounded values: tensor([2.1875, 2.2031], device='cuda:0', dtype=torch.bfloat16)
Kernel's probability of rounding up: 0.8003
Expected probability: 0.8011
ok
test_rounding_statistics_2 (__main__.TestStochasticRounding.test_rounding_statistics_2)
Test stochastic rounding with different BF16 boundary values ...
Input value: 1.7999992370605469
Possible rounded values: tensor([1.7969, 1.8047], device='cuda:0', dtype=torch.bfloat16)
Kernel's probability of rounding up: 0.3994
Expected probability: 0.3973
ok
test_rounding_statistics_large (__main__.TestStochasticRounding.test_rounding_statistics_large)
Test stochastic rounding for large number, over 100 ...
Input value: 128.99998474121094
Kernel's probability of rounding up: 1.0000
Expected probability: 0.9999
ok
test_rounding_statistics_small (__main__.TestStochasticRounding.test_rounding_statistics_small)
Test stochastic rounding for number between 0 and 1 ...
Input value: 0.7499847412109375
Kernel's probability of rounding up: 0.9961
Expected probability: 0.9924
ok
test_small_values (__main__.TestStochasticRounding.test_small_values)
Test handling of small values near zero ... ok
test_special_values (__main__.TestStochasticRounding.test_special_values)
Test handling of special values like inf, -inf, nan ... ok
test_vectorized_loading (__main__.TestStochasticRounding.test_vectorized_loading)
Test if vectorized loading works correctly for different tensor sizes ... ok

----------------------------------------------------------------------
Ran 8 tests in 1.343s

OK
