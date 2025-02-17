#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <random>
#include <torch/extension.h>

// Forward declaration of CUDA kernel
extern "C" __global__ void
stochastic_round_bf16(float *__restrict__ input,
                      __nv_bfloat16 *__restrict__ output, const int size,
                      const unsigned long long seed);

// Get optimal block size - declaration only
__host__ int getOptimalBlockSize();

// C++ wrapper for the CUDA kernel - declaration only
torch::Tensor stochastic_round_bf16_cuda(torch::Tensor input);
