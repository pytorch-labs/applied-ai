#include "stochastic_rounding.hpp"
#include <cstdint>

#define PHILOX_ROUNDS 7 // per Natalia

__device__ __forceinline__ __nv_bfloat16
float_to_bf16_stochastic(float value, uint16_t rand16) {
  // Handle special cases first
  // TODO - is this the right response..just return?
  if (__isnanf(value) || __isinff(value)) {
    return __float2bfloat16(value);
  }

  uint32_t value_uint32 = __float_as_uint(value);
  value_uint32 = (value_uint32 + rand16) & 0xFFFF0000u;
  return __float2bfloat16(__uint_as_float(value_uint32));
}

// Main kernel
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
    // Generate 4 random numbers at once
    float4 rand4 = curand_uniform4(&state);
    uint32_t *rand_bits = (uint32_t *)&rand4;

    // Process up to 4 elements
    for (int offset = 0; offset < 4 && base_idx + offset < size; offset++) {

      uint16_t rand16_low = rand_bits[offset] & 0xFFFF;
      uint16_t rand16_high = (rand_bits[offset] >> 16) & 0xFFFF;

      // We process two elements if possible per random generation
      // First
      int curr_idx = base_idx + offset;
      output[curr_idx] = float_to_bf16_stochastic(input[curr_idx], rand16_low);

      // Second element if within bounds
      int next_idx = curr_idx + 4;
      if (next_idx < size) {
        output[next_idx] =
            float_to_bf16_stochastic(input[next_idx], rand16_high);
      }
    }
  }
}
