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

// Get optimal block size
__host__ int getOptimalBlockSize() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  return std::min(prop.maxThreadsPerBlock, 256);
}

// C++ wrapper for the CUDA kernel
torch::Tensor stochastic_round_bf16_cuda(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
  TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
  TORCH_CHECK(input.scalar_type() == torch::kFloat32,
              "Input tensor must be float32");

  const int threads_per_block = getOptimalBlockSize();
  const int num_elements = input.numel();
  const int blocks = (num_elements + threads_per_block - 1) / threads_per_block;

  // Check maximum grid dimension
  constexpr int MAX_BLOCKS = 65535;
  TORCH_CHECK(blocks <= MAX_BLOCKS,
              "Input size too large. Maximum supported size is ",
              MAX_BLOCKS * threads_per_block, " elements");

  // Create output tensor
  auto options = torch::TensorOptions()
                     .dtype(torch::kBFloat16)
                     .device(input.device())
                     .requires_grad(false);
  auto output = torch::empty_like(input, options);

  // Generate random seed
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<unsigned long long> dis;
  const unsigned long long seed = dis(gen);

  // Launch kernel
  stochastic_round_bf16<<<blocks, threads_per_block>>>(
      input.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16 *>(output.data_ptr()), num_elements, seed);

  // Check for CUDA errors
  TORCH_CHECK(cudaGetLastError() == cudaSuccess,
              "CUDA kernel execution failed");

  return output;
}
