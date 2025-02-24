 #include "stochastic_rounding.hpp"
#include <cstdint>

// Philox RNG implementation

__device__ __forceinline__ PhiloxGenerator::PhiloxGenerator() :
    key(make_uint2(0, 0)),
    counter(make_uint4(0, 0, 0, 0)) {}

__device__ __forceinline__ void PhiloxGenerator::init(const unsigned long long seed, const unsigned int thread_id) {
    key.x = static_cast<unsigned int>(seed);
    key.y = static_cast<unsigned int>(seed >> 32);
    counter = make_uint4(thread_id, 0, 0, 0);
    __threadfence_block();
}

__device__ __forceinline__ uint2 PhiloxGenerator::mulhilo(const unsigned int a, const unsigned int b) {
    uint2 result;
    unsigned long long prod;
    asm("mul.wide.u32 %0, %1, %2;" : "=l"(prod) : "r"(a), "r"(b));
    result.x = static_cast<unsigned int>(prod);
    result.y = static_cast<unsigned int>(prod >> 32);
    return result;
}

__device__ __forceinline__ uint4 PhiloxGenerator::round(uint4 ctr, uint2 key) {
    const uint2 mul0 = mulhilo(philox::M0, ctr.x);
    const uint2 mul1 = mulhilo(philox::M1, ctr.z);

    return make_uint4(
        mul1.y ^ ctr.y ^ key.x,
        mul1.x,
        mul0.y ^ ctr.w ^ key.y,
        mul0.x
    );
}

__device__ __forceinline__ uint4 PhiloxGenerator::next() {
    uint4 ctr = counter;
    uint2 k = key;

    #pragma unroll
    for (int i = 0; i < philox::ROUNDS; ++i) {
        ctr = round(ctr, k);
        k.x += philox::W32_0;
        k.y += philox::W32_1;
    }

    counter.x += 4;
    return ctr;
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16_stochastic(const float value, const uint32_t rand) {
    const uint32_t val_bits = __float_as_uint(value);
    const uint32_t rounding_bits = val_bits & 0xFFFF;
    uint32_t result = val_bits & 0xFFFF0000u;
    result += (rand & 0xFFFF) < rounding_bits ? 0x10000u : 0;
    return __float2bfloat16(__uint_as_float(result));
}

__device__ __forceinline__ void float4_to_bf16_stochastic(
    const float4& values,
    uint4& rand_vals,
    __nv_bfloat16* output) {

    float vals[4] = {values.x, values.y, values.z, values.w};
    uint32_t rands[4] = {rand_vals.x, rand_vals.y, rand_vals.z, rand_vals.w};

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        output[i] = float_to_bf16_stochastic(vals[i], rands[i]);
    }
}

__global__ void stochastic_round_bf16(
    float *__restrict__ input,
    __nv_bfloat16 *__restrict__ output,
    const int size,
    const unsigned long long seed) {

    PhiloxGenerator rng;
    rng.init(seed, blockIdx.x * blockDim.x + threadIdx.x);

    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int stride = blockDim.x * gridDim.x * 4;

    float4 values;
    __nv_bfloat16 local_output[4];

    // Process full vectors of 4 elements
    for (; idx <= size - 4; idx += stride) {
        values = *reinterpret_cast<float4*>(&input[idx]);
        uint4 rand = rng.next();
        float4_to_bf16_stochastic(values, rand, local_output);

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

        uint4 rand = rng.next();
        float4_to_bf16_stochastic(values, rand, local_output);

        for (int j = 0; j < remainder; j++) {
            output[idx + j] = local_output[j];
        }
    }
}
