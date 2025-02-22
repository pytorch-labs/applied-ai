#include "stochastic_rounding.hpp"
#include <cstdint>

namespace cg = cooperative_groups;

namespace {

// Philox implementation
class Philox {
public:
    __device__ inline Philox() : STATE(0) {
    key = {0, 0};
    counter = {0, 0, 0, 0};
}
    struct alignas(16) ull2 {  // Ensure 16-byte alignment
        uint64_t x;
        uint64_t y;
    };

    __device__ inline Philox(unsigned long long seed,
                            unsigned long long subsequence,
                            unsigned long long offset) : STATE(0) {
        key = reinterpret_cast<const uint2&>(seed);
        ull2* tmp = reinterpret_cast<ull2*>(&counter);
        tmp->x = offset / 4;
        tmp->y = subsequence;
    }

    __device__ inline uint4 operator()() {
        uint4 counter_ = counter;
        uint2 key_ = key;

        #pragma unroll
        for (int i = 0; i < 6; i++) {
            counter_ = single_round(counter_, key_);
            key_.x += (kPhilox10A);
            key_.y += (kPhilox10B);
        }
        output = single_round(counter_, key_);
        counter = incr128(counter);
        return output;
    }

private:
    alignas(16) uint4 counter;  // Ensure 16-byte alignment
    alignas(16) uint4 output;   // Ensure 16-byte alignment
    uint2 key;
    unsigned int STATE;

    static __device__ __forceinline__ uint4 incr128(uint4 ctr) {
        uint4 res;
        asm ("add.cc.u32      %0, %4, %8;\n\t"
             "addc.cc.u32     %1, %5, %9;\n\t"
             "addc.cc.u32     %2, %6, %10;\n\t"
             "addc.u32        %3, %7, %11;\n\t"
             : "=r"(res.x), "=r"(res.y), "=r"(res.z), "=r"(res.w)
             : "r"(ctr.x), "r"(ctr.y), "r"(ctr.z), "r"(ctr.w),
               "n"(1), "n"(0), "n"(0), "n"(0));
        return res;
    }

    static __device__ __forceinline__ uint2 mulhilo32_v2(const unsigned int a, const unsigned int b) {
        uint2 *res;
        unsigned long long tmp;
        asm ("mul.wide.u32      %0, %1, %2;\n\t"
             : "=l"(tmp)
             : "r"(a), "r"(b));
        res = (uint2*)(&tmp);
        return *res;
    }

    static __device__ __forceinline__ uint4 single_round(const uint4 ctr, const uint2 key) {
        uint2 res0 = mulhilo32_v2(kPhiloxSA, ctr.x);
        uint2 res1 = mulhilo32_v2(kPhiloxSB, ctr.z);
        uint4 ret = {res1.y ^ ctr.y ^ key.x, res1.x, res0.y ^ ctr.w ^ key.y, res0.x};
        return ret;
    }

    static const unsigned long kPhilox10A = 0x9E3779B9;
    static const unsigned long kPhilox10B = 0xBB67AE85;
    static const unsigned long kPhiloxSA = 0xD2511F53;
    static const unsigned long kPhiloxSB = 0xCD9E8D57;
};

// Vectorized stochastic rounding helper
__device__ __forceinline__ void float4_to_bf16x4_stochastic(
    const float4& values,
    __nv_bfloat162& out_low,
    __nv_bfloat162& out_high,
    const uint4& rand4) {

    // Process first pair with minimal register usage
    const uint32_t val1_bits = __float_as_uint(values.x);
    const uint32_t val2_bits = __float_as_uint(values.y);

    uint32_t round1 = val1_bits & 0xFFFF0000u;
    uint32_t round2 = val2_bits & 0xFFFF0000u;

    round1 += ((rand4.x & 0xFFFF) < (val1_bits & 0xFFFF)) ? 0x10000 : 0;
    round2 += ((rand4.y & 0xFFFF) < (val2_bits & 0xFFFF)) ? 0x10000 : 0;

    const __nv_bfloat16 bf16_1 = __float2bfloat16(__uint_as_float(round1));
    const __nv_bfloat16 bf16_2 = __float2bfloat16(__uint_as_float(round2));
    out_low = __nv_bfloat162(bf16_1, bf16_2);

    // Process second pair
    const uint32_t val3_bits = __float_as_uint(values.z);
    const uint32_t val4_bits = __float_as_uint(values.w);

    round1 = val3_bits & 0xFFFF0000u;
    round2 = val4_bits & 0xFFFF0000u;

    round1 += ((rand4.z & 0xFFFF) < (val3_bits & 0xFFFF)) ? 0x10000 : 0;
    round2 += ((rand4.w & 0xFFFF) < (val4_bits & 0xFFFF)) ? 0x10000 : 0;

    const __nv_bfloat16 bf16_3 = __float2bfloat16(__uint_as_float(round1));
    const __nv_bfloat16 bf16_4 = __float2bfloat16(__uint_as_float(round2));
    out_high = __nv_bfloat162(bf16_3, bf16_4);
}

} // namespace

// Main kernel
extern "C" __global__ void __launch_bounds__(256, 4) stochastic_round_bf16(
    const float4 *__restrict__ input,
    __nv_bfloat162 *__restrict__ output,
    const int n_vec4,
    const unsigned long long seed) {

    auto block = cg::this_thread_block();
    const auto warp = cg::tiled_partition<32>(block);

    __shared__ alignas(16) Philox block_rng;

    if (block.thread_rank() == 0) {
        new (&block_rng) Philox(seed + blockIdx.x, 0, 0);
    }
    block.sync();

    // Grid-stride loop with vectorized processing
    const int vec4_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int vec4_stride = gridDim.x * blockDim.x;
    const int thread_in_group = threadIdx.x % 4;
    const int warp_lane = warp.thread_rank();

    #pragma unroll 1
    for (int i = vec4_idx; i < n_vec4; i += vec4_stride) {
        uint4 rand4;
        if (thread_in_group == 0) {
            rand4 = block_rng();
        }

        // Use faster warp shuffle for random number distribution
        rand4.x = __shfl_sync(0xffffffff, rand4.x, (warp_lane/4)*4, 4);
        rand4.y = __shfl_sync(0xffffffff, rand4.y, (warp_lane/4)*4, 4);
        rand4.z = __shfl_sync(0xffffffff, rand4.z, (warp_lane/4)*4, 4);
        rand4.w = __shfl_sync(0xffffffff, rand4.w, (warp_lane/4)*4, 4);

        if (i < n_vec4) {  // Boundary check
            const float4 values = input[i];
            __nv_bfloat162 out_low, out_high;
            float4_to_bf16x4_stochastic(values, out_low, out_high, rand4);

            // Coalesced stores
            output[i * 2] = out_low;
            output[i * 2 + 1] = out_high;
        }
    }
}
