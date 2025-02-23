#include "stochastic_rounding.hpp"
#include <cstdint>

#define PHILOX_ROUNDS 7

__device__ __forceinline__ void float4_to_bf16_stochastic(
    const float4& values,
    uint4& rand_vals,
    __nv_bfloat16* output) {

    float vals[4] = {values.x, values.y, values.z, values.w};
    uint32_t rands[4] = {rand_vals.x, rand_vals.y, rand_vals.z, rand_vals.w};

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint32_t value_bits = __float_as_uint(vals[i]);
        uint32_t truncated = value_bits & 0xFFFF;
        uint32_t rounded = value_bits & 0xFFFF0000u;

        bool should_round_up = (rands[i] & 0xFFFF) < truncated;
        if (should_round_up) {
            rounded += 0x10000;
        }

        output[i] = __float2bfloat16(__uint_as_float(rounded));
    }
}

extern "C" __global__ void stochastic_round_bf16(
    float *__restrict__ input,
    __nv_bfloat16 *__restrict__ output,
    const int size,
    const unsigned long long seed) {

    curandStatePhilox4_32_10_t state;
    // Initialize with sequence number as offset
    curand_init(seed + clock64(), blockIdx.x * blockDim.x + threadIdx.x, 0, &state);


    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int stride = blockDim.x * gridDim.x * 4;

    float4 values;
    __nv_bfloat16 local_output[4];

    // Process full vectors of 4 elements
    for (; idx <= size - 4; idx += stride) {
        // Load 4 consecutive float values
        values = *reinterpret_cast<float4*>(&input[idx]);

        // Generate 4 random numbers at once
        uint4 rand = curand4(&state);

        // Convert and round all 4 values
        float4_to_bf16_stochastic(values, rand, local_output);

        // Store results
        for (int j = 0; j < 4; j++) {
            output[idx + j] = local_output[j];
        }
    }

    // Handle remaining elements
    if (idx < size) {
        float remaining_values[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        int remainder = size - idx;

        for (int j = 0; j < remainder; j++) {
            remaining_values[j] = input[idx + j];
        }

        values.x = remaining_values[0];
        values.y = remaining_values[1];
        values.z = remaining_values[2];
        values.w = remaining_values[3];

        uint4 rand = curand4(&state);
        float4_to_bf16_stochastic(values, rand, local_output);

        for (int j = 0; j < remainder; j++) {
            output[idx + j] = local_output[j];
        }
    }
}
