#include "stochastic_rounding.hpp"
#include <cstdint>

#define PHILOX_ROUNDS 7 // Per Natalia

__device__ __forceinline__ __nv_bfloat16
float_to_bf16_stochastic(float value, uint32_t rand32) {
    uint32_t value_bits = __float_as_uint(value);
    uint32_t truncated = value_bits & 0xFFFF;  // Get lower 16 bits
    uint32_t rounded = value_bits & 0xFFFF0000u;

    bool should_round_up = (rand32 & 0xFFFF) < truncated;

    if (should_round_up) {
        rounded += 0x10000;
    }

    return __float2bfloat16(__uint_as_float(rounded));
}

// stochastic_round_bf16 Main kernel
__global__ void stochastic_round_bf16(float *__restrict__ input,
                                     __nv_bfloat16 *__restrict__ output,
                                     const int size,
                                     const unsigned long long seed) {
    curandStatePhilox4_32_10_t state;
    curand_init(seed + clock64(), blockIdx.x * blockDim.x + threadIdx.x, 0, &state);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size; i += stride) {
        uint32_t rand_val = curand(&state);
        output[i] = float_to_bf16_stochastic(input[i], rand_val);
    }
}
