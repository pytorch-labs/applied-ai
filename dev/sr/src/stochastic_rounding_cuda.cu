#include "stochastic_rounding.hpp"
#include <cstdint>

// Performance-optimized utility function
__device__ __forceinline__ __nv_bfloat16
float_to_bf16_stochastic(float x, curandState *state) {
  // Use intrinsics for special case checking
  if (__isnanf(x) || __isinff(x)) {
    return __float2bfloat16(x);
  }

  int x_int = __float_as_int(x);
  // Extract components
  uint32_t sign = x_int & 0x80000000;
  uint32_t exp = x_int & 0x7F800000;
  uint32_t mant = x_int & 0x007FFFFF;

  // Handle special case - zero
  if (exp == 0 && mant == 0) {
    return __float2bfloat16(x);
  }

  // Calculate rounding probability using upper 8 bits of the 16 dropped bits
  uint32_t upper_dropped_bits = (mant & 0x0000FF00) >> 8;
  // Add epsilon to avoid potential division by zero
  const float EPSILON = 1e-7f;
  float prob = float(upper_dropped_bits) / (float(0x100) + EPSILON);

  // Use the pre-initialized random state
  float rand = curand_uniform(state);

  // Round up if random number is less than probability
  uint32_t round_bit = (rand < prob) ? 0x80 : 0;

  // Combine components with stochastic rounding
  uint32_t bf16_bits = sign | exp | ((mant + (round_bit << 8)) >> 16);

  // Convert back using intrinsic
  return __float2bfloat16(__int_as_float(bf16_bits));
}

// Implementation of the kernel
__global__ void stochastic_round_bf16(float *__restrict__ input,
                                      __nv_bfloat16 *__restrict__ output,
                                      const int size,
                                      const unsigned long long seed) {
  // Initialize random state once per thread
  curandState state;
  curand_init(seed, blockIdx.x * blockDim.x + threadIdx.x, 0, &state);

  // Process multiple elements with the same random state
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += blockDim.x * gridDim.x) {
    output[idx] = float_to_bf16_stochastic(input[idx], &state);
  }
}
