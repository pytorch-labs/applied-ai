#include "stochastic_rounding.hpp"
#include <cstdint>

// Performance-optimized utility function
__device__ __forceinline__ __nv_bfloat16
float_to_bf16_stochastic(float x, curandState *state) {
  // Use intrinsics for special case checking
  if (__isnanf(x) || __isinff(x)) {
    return __float2bfloat16(x);
  }

  uint32_t x_int = __float_as_uint(x);

  // Extract sign and exponent (these stay the same)
  uint32_t sign_exp = x_int & 0xFF800000;

  // Extract mantissa
  uint32_t mantissa = x_int & 0x007FFFFF;

  // Handle special case - zero or denormal
  if (sign_exp == 0) {
    return __float2bfloat16(x);
  }

  // Take the lowest 16 bits for probability
  // If these bits are higher, we're more likely to round up
  uint32_t prob_bits = mantissa & 0x0000FFFF;

  // Generate random number (0-65535)
  uint32_t rand = (uint32_t)(curand_uniform(state) * 65536.0f);

  // If random number is less than prob_bits, round up the 7th mantissa bit
  uint32_t rounded = mantissa >> 16; // Keep top 7 bits
  if (rand < prob_bits) {
    rounded += 1;
    // Handle carry to exponent
    if (rounded == 0x80) {
      rounded = 0;
      sign_exp += 0x00800000;
    }
  }

  // Combine components
  uint32_t result = sign_exp | rounded;

  return __float2bfloat16(__uint_as_float(result));
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
