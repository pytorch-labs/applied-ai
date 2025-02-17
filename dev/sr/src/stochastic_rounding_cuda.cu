#include "stochastic_rounding.hpp"
#include <cstdint>

#define PHILOX_ROUNDS 7 // Per Natalia

__device__ __forceinline__ __nv_bfloat16
float_to_bf16_stochastic(float value, uint32_t rand32) {
  // Handle special cases first - TODO - is returning the correct response?
  if (__isnanf(value) || __isinff(value)) {
    return __float2bfloat16(value);
  }

  uint32_t value_uint32 = __float_as_uint(value);

  // Extract the truncated bits (lower 16 bits of mantissa)
  uint32_t truncated = value_uint32 & 0xFFFF;

  // If random value is less than truncated bits, round up
  // This ensures proper probability mapping
  // e.g., if truncated = 0xB333 ≈ 0.7 * 0xFFFF,
  // then probability of rounding up is ~0.7
  uint32_t rounded = value_uint32 & 0xFFFF0000u;
  if (rand32 < truncated) {
    rounded += 0x10000; // Round up
  }

  return __float2bfloat16(__uint_as_float(rounded));
}

// stochastic_round_bf16 Main kernel
__global__ void stochastic_round_bf16(float *__restrict__ input,
                                      __nv_bfloat16 *__restrict__ output,
                                      const int size,
                                      const unsigned long long seed) {

  // Initialize Philox state for this thread
  curandStatePhilox4_32_10_t state;
  curand_init(seed, blockIdx.x * blockDim.x + threadIdx.x, PHILOX_ROUNDS,
              &state);

  // Process elements in groups of 4
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int base_idx = idx * 4; base_idx < size; base_idx += stride * 4) {

    float4 rand4 = curand_uniform4(&state);
    uint32_t *rand_bits = (uint32_t *)&rand4;

    // Process up to 4 elements
    for (int offset = 0; offset < 4 && base_idx + offset < size; offset++) {
      // Get two random 16-bit values
      uint32_t rand_low = rand_bits[offset] & 0xFFFF;
      uint32_t rand_high = rand_bits[offset] >> 16;

      // Process first element
      int curr_idx = base_idx + offset;
      output[curr_idx] = float_to_bf16_stochastic(input[curr_idx], rand_low);

      // Process second element if within bounds
      int next_idx = curr_idx + 4;
      if (next_idx < size) {
        output[next_idx] = float_to_bf16_stochastic(input[next_idx], rand_high);
      }
    }
  }
}
