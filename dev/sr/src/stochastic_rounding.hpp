
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>

namespace philox {
    constexpr unsigned int W32_0   = 0x9E3779B9;
    constexpr unsigned int W32_1   = 0xBB67AE85;
    constexpr unsigned int M0      = 0xD2511F53;
    constexpr unsigned int M1      = 0xCD9E8D57;
    constexpr int ROUNDS          = 7;
}

// Forward declarations
class PhiloxGenerator {
public:
    __device__ __forceinline__ PhiloxGenerator();
    __device__ __forceinline__ void init(const unsigned long long seed, const unsigned int thread_id);
    __device__ __forceinline__ uint4 next();
private:
    uint2 key;
    uint4 counter;
    static __device__ __forceinline__ uint2 mulhilo(const unsigned int a, const unsigned int b);
    static __device__ __forceinline__ uint4 round(uint4 ctr, uint2 key);
};

// CUDA kernel declaration
__global__ void stochastic_round_bf16(
    float *__restrict__ input,
    __nv_bfloat16 *__restrict__ output,
    const int size,
    const unsigned long long seed);

// Host functions
__host__ int getOptimalBlockSize();
torch::Tensor stochastic_round_bf16_cuda(torch::Tensor input, bool requires_grad = false);
