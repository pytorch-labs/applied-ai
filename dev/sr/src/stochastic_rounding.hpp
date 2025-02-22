#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <random>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <cooperative_groups.h>

// Forward declarations
extern "C" __global__ void stochastic_round_bf16(
    const float4 *__restrict__ input,
    __nv_bfloat162 *__restrict__ output,
    const int n_vec4,
    const unsigned long long seed);

torch::Tensor stochastic_round_bf16_cuda(torch::Tensor input, bool requires_grad = false);
