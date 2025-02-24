#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <cstdint>
#include <torch/extension.h>
#include <pybind11/pybind11.h>

namespace philox {
    constexpr uint32_t W32_0   = 0x9E3779B9;
    constexpr uint32_t W32_1   = 0xBB67AE85;
    constexpr uint32_t M0      = 0xD2511F53;
    constexpr uint32_t M1      = 0xCD9E8D57;
    constexpr int ROUNDS       = 7;
}

// Forward declarations
class PhiloxGenerator {
public:
    __device__ __forceinline__ PhiloxGenerator();
    __device__ __forceinline__ void init(uint64_t seed, uint32_t thread_id);
    __device__ __forceinline__ uint4 next();
private:
    uint2 key;
    uint4 counter;
    static __device__ __forceinline__ uint2 mulhilo(uint32_t a, uint32_t b);
    static __device__ __forceinline__ uint4 round(uint4 ctr, uint2 key);
};

// CUDA kernel declarations
__global__ void stochastic_round_bf16(
    float *__restrict__ input,
    __nv_bfloat16 *__restrict__ output,
    int size,
    uint64_t seed);

__global__ void stochastic_round_fp16(
    float *__restrict__ input,
    __half *__restrict__ output,
    int size,
    uint64_t seed);

// Host functions
__host__ int getOptimalBlockSize();

// PyTorch wrapper functions
torch::Tensor stochastic_round_bf16_cuda(torch::Tensor input, bool requires_grad = false);
torch::Tensor stochastic_round_fp16_cuda(torch::Tensor input, bool requires_grad = false);
