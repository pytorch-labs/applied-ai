#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <vector_types.h>  // for uint2, uint4
#include <random>
#include <torch/extension.h>
#include <pybind11/pybind11.h>

// Philox RNG constants
namespace philox_constants {
    constexpr unsigned int PHILOX_W32_0   = 0x9E3779B9;
    constexpr unsigned int PHILOX_W32_1   = 0xBB67AE85;
    constexpr unsigned int PHILOX_M0      = 0xD2511F53;
    constexpr unsigned int PHILOX_M1      = 0xCD9E8D57;
}

// Vector types declarations
struct alignas(16) float8 {
    float x[8];
};

struct alignas(8) float2_aligned {
    float x[2];
};

// Philox RNG class declaration
class Philox {
public:
    __device__ __forceinline__ Philox();
    __device__ __forceinline__ void init(const unsigned long long seed, const unsigned int thread_id);
    __device__ __forceinline__ uint4 operator()();
private:
    uint2 key;
    uint4 counter;
    static __device__ __forceinline__ uint2 mulhilo32(const unsigned int a, const unsigned int b);
    static __device__ __forceinline__ uint4 single_round(const uint4 ctr, const uint2 key);
};

// BF16 conversion helper
__device__ __forceinline__ __nv_bfloat16 float_to_bf16_stochastic(float value, uint32_t rand32);

// Forward declaration of helper device function
__device__ __forceinline__ void float4_to_bf16_stochastic(
    const float4& values,
    uint4& rand_vals,
    __nv_bfloat16* output);

// Forward declaration of CUDA kernel
__global__ void stochastic_round_bf16(
    float *__restrict__ input,
    __nv_bfloat16 *__restrict__ output,
    const int size,
    const unsigned long long seed);

// Get optimal block size - declaration only
__host__ int getOptimalBlockSize();



// C++ wrapper for the CUDA kernel - declaration only
torch::Tensor stochastic_round_bf16_cuda(torch::Tensor input, bool requires_grad = false);
