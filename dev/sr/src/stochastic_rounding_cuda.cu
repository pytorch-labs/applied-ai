#include "stochastic_rounding.hpp"
#include <cmath>

__device__ __forceinline__ PhiloxGenerator::PhiloxGenerator() :
    key(make_uint2(0, 0)),
    counter(make_uint4(0, 0, 0, 0)) {}

__device__ __forceinline__ void PhiloxGenerator::init(uint64_t seed, uint32_t thread_id) {
    key.x = static_cast<uint32_t>(seed);
    key.y = static_cast<uint32_t>(seed >> 32);
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

// BF16 stochastic rounding - 16 bits total, with 7 bits mantissa
__device__ __forceinline__ __nv_bfloat16 float_to_bf16_stochastic(const float value, const uint32_t rand) {
    // Handle special cases first
    if (!isfinite(value)) {
        return __float2bfloat16(value);
    }

    // Extract bits from float
    const uint32_t val_bits = __float_as_uint(value);

    // For BF16, we keep top 16 bits and use next 8 bits for rounding
    const uint32_t truncated = val_bits & 0xFFFF0000u;  // Top 16 bits
    const uint32_t rounding_bits = (val_bits >> 16) & 0xFF;  // Next 8 bits

    // Round up if random value is less than truncated bits
    const uint32_t random_bits = (rand & 0xFF);
    const uint32_t rounded = truncated + (random_bits < rounding_bits ? 0x10000u : 0);

    return __float2bfloat16(__uint_as_float(rounded));
}

// FP16 stochastic rounding - 16 bits total, with 10 bits mantissa
__device__ __forceinline__ __half float_to_fp16_stochastic(const float value, const uint32_t rand) {
    // Handle special cases first
    if (!isfinite(value)) {
        return __float2half(value);
    }

    const uint32_t val_bits = __float_as_uint(value);
    const uint32_t sign = val_bits & 0x80000000u;  // Extract sign bit
    const uint32_t exp = (val_bits >> 23) & 0xFFu;  // Extract exponent
    const uint32_t mant = val_bits & 0x7FFFFFu;     // Extract mantissa

    // Handle subnormals and exponent adjustment
    if (exp == 0) {
        // Input is subnormal or zero
        return __float2half(value);
    }

    // Adjust exponent bias from FP32 (127) to FP16 (15)
    const int new_exp = exp - 127 + 15;

    if (new_exp < 0) {
        // Result would be subnormal in FP16
        return __float2half(value);  // Let CUDA handle subnormal conversion
    }

    if (new_exp > 31) {
    // Would overflow FP16's exponent range
    return __float2half(sign ? -INFINITY : INFINITY);
}

    // FP16 mantissa is 10 bits, so we need to round from 23 to 10 bits
    const uint32_t mant_rounding_bits = mant & 0x1FFFu;  // Bottom 13 bits
    const uint32_t mant_msb = mant >> 13;               // Top 10 bits

    // Generate probability from the truncated bits
    const uint32_t random = rand & 0x1FFFu;  // Use 13 bits of randomness

    // Round up if random value is less than truncated bits
    const uint32_t round_up = (random < mant_rounding_bits) ? 1u : 0u;

    // Combine the pieces into FP16's bit pattern
    // [15] sign bit
    // [14:10] exponent (5 bits)
    // [9:0] mantissa (10 bits)
    const uint16_t h_bits = ((sign >> 16) & 0x8000u) |                 // Sign bit
                           ((new_exp & 0x1Fu) << 10) |        // Exponent
                           ((mant_msb + round_up) & 0x3FFu);  // Mantissa with rounding

    __half result;
    __half_raw* raw_ptr = reinterpret_cast<__half_raw*>(&result);
    raw_ptr->x = h_bits;
    return result;
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

__device__ __forceinline__ void float4_to_fp16_stochastic(
    const float4& values,
    uint4& rand_vals,
    __half* output) {

    float vals[4] = {values.x, values.y, values.z, values.w};
    uint32_t rands[4] = {rand_vals.x, rand_vals.y, rand_vals.z, rand_vals.w};

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        output[i] = float_to_fp16_stochastic(vals[i], rands[i]);
    }
}

__global__ void stochastic_round_bf16(
    float *__restrict__ input,
    __nv_bfloat16 *__restrict__ output,
    int size,
    uint64_t seed) {

    PhiloxGenerator rng;
    rng.init(seed, blockIdx.x * blockDim.x + threadIdx.x);

    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int stride = blockDim.x * gridDim.x * 4;

    float4 values;
    __nv_bfloat16 local_output[4];

    for (; idx <= size - 4; idx += stride) {
        values = *reinterpret_cast<float4*>(&input[idx]);
        uint4 rand = rng.next();
        float4_to_bf16_stochastic(values, rand, local_output);

        for (int j = 0; j < 4; j++) {
            output[idx + j] = local_output[j];
        }
    }

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

__global__ void stochastic_round_fp16(
    float *__restrict__ input,
    __half *__restrict__ output,
    int size,
    uint64_t seed) {

    PhiloxGenerator rng;
    rng.init(seed, blockIdx.x * blockDim.x + threadIdx.x);

    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int stride = blockDim.x * gridDim.x * 4;

    float4 values;
    __half local_output[4];

    for (; idx <= size - 4; idx += stride) {
        values = *reinterpret_cast<float4*>(&input[idx]);
        uint4 rand = rng.next();
        float4_to_fp16_stochastic(values, rand, local_output);

        for (int j = 0; j < 4; j++) {
            output[idx + j] = local_output[j];
        }
    }

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
        float4_to_fp16_stochastic(values, rand, local_output);

        for (int j = 0; j < remainder; j++) {
            output[idx + j] = local_output[j];
        }
    }
}
