#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <random>
#include <torch/extension.h>
#include <pybind11/pybind11.h>


// Forward declaration of CUDA kernel
extern "C" __global__ void
stochastic_round_bf16(float *__restrict__ input,
                      __nv_bfloat16 *__restrict__ output, const int size,
                      const unsigned long long seed);

// Get optimal block size - declaration only
__host__ int getOptimalBlockSize();

// C++ wrapper for the CUDA kernel - declaration only
//torch::Tensor stochastic_round_bf16_cuda(torch::Tensor input);

// Architecture-specific configuration
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
// Hopper-specific configurations
#define OPTIMAL_BLOCK_SIZE 256
#define VECTOR_SIZE 4
#define ELEMENTS_PER_THREAD 8
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
// Ampere-specific configurations
#define OPTIMAL_BLOCK_SIZE 256
#define VECTOR_SIZE 4
#define ELEMENTS_PER_THREAD 8
#else
// Default configurations
#define OPTIMAL_BLOCK_SIZE 256
#define VECTOR_SIZE 4
#define ELEMENTS_PER_THREAD 4
#endif
