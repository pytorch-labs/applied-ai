 #include "stochastic_rounding.hpp"
#include <cstdint>

// Philox RNG implementation
__device__ __forceinline__ Philox::Philox() : key(make_uint2(0, 0)), counter(make_uint4(0, 0, 0, 0)) {}

__device__ __forceinline__ void Philox::init(const unsigned long long seed, const unsigned int thread_id) {
    key = *reinterpret_cast<const uint2*>(&seed);
    counter = make_uint4(thread_id, 0, 0, 0);
    __threadfence_block();
}

__device__ __forceinline__ uint2 Philox::mulhilo32_v2(const unsigned int a, const unsigned int b) {
    unsigned long long tmp;
    asm ("mul.wide.u32 %0, %1, %2;\n\t"
         : "=l"(tmp)
         : "r"(a), "r"(b));
    return *reinterpret_cast<uint2*>(&tmp);
}

__device__ __forceinline__ uint4 Philox::single_round(const uint4 ctr, const uint2 key) {
    using namespace philox_constants;
    const uint2 res0 = mulhilo32_v2(PHILOX_M0, ctr.x);
    const uint2 res1 = mulhilo32_v2(PHILOX_M1, ctr.z);
    return make_uint4(res1.y ^ ctr.y ^ key.x, res1.x,
                     res0.y ^ ctr.w ^ key.y, res0.x);
}

__device__ __forceinline__ uint4 Philox::operator()() {
    using namespace philox_constants;
    uint4 counter_ = counter;
    uint2 key_ = key;

    #pragma unroll
    for (int i = 0; i < 7; i++) {
        counter_ = single_round(counter_, key_);
        key_.x += PHILOX_W32_0;
        key_.y += PHILOX_W32_1;
    }
    counter.x += 4;
    return counter_;
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16_stochastic(float value, uint32_t rand32) {
    uint32_t value_bits = __float_as_uint(value);
    uint32_t truncated = value_bits & 0xFFFF;
    uint32_t rounded = value_bits & 0xFFFF0000u;

    bool should_round_up = (rand32 & 0xFFFF) < truncated;
    if (should_round_up) {
        rounded += 0x10000;
    }

    return __float2bfloat16(__uint_as_float(rounded));
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

    // Initialize Philox RNG
    Philox rng;
    rng.init(seed, blockIdx.x * blockDim.x + threadIdx.x);

    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int stride = blockDim.x * gridDim.x * 4;

    float4 values;
    __nv_bfloat16 local_output[4];

    // Process full vectors of 4 elements
    for (; idx <= size - 4; idx += stride) {
        // Load 4 consecutive float values
        values = *reinterpret_cast<float4*>(&input[idx]);

        // Generate 4 random numbers using Philox
        uint4 rand = rng();

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

        uint4 rand = rng();
        float4_to_bf16_stochastic(values, rand, local_output);

        for (int j = 0; j < remainder; j++) {
            output[idx + j] = local_output[j];
        }
    }
}
