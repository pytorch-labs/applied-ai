
#include <pybind11/pybind11.h>
#include "stochastic_rounding.hpp"
#include <random>

namespace py = pybind11;

__host__ int getOptimalBlockSize() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    return std::min(prop.maxThreadsPerBlock, 256);
}

torch::Tensor stochastic_round_bf16_cuda(torch::Tensor input, bool requires_grad) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input tensor must be float32");

    const int threads_per_block = 256;
    const int num_elements = input.numel();
    const int elements_per_thread = 4;

    const int min_blocks = (num_elements + elements_per_thread * threads_per_block - 1) /
                          (elements_per_thread * threads_per_block);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const int blocks_per_sm = 4;
    const int min_blocks_for_sms = prop.multiProcessorCount * blocks_per_sm;
    const int num_blocks = std::max(min_blocks, min_blocks_for_sms);

    auto options = torch::TensorOptions()
                      .dtype(torch::kBFloat16)
                      .device(input.device())
                      .requires_grad(requires_grad);
    auto output = torch::empty_like(input, options);

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<unsigned long long> dis;
    const unsigned long long seed = dis(gen);

    stochastic_round_bf16<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        num_elements,
        seed);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "CUDA kernel execution failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("stochastic_round_bf16",
          static_cast<torch::Tensor (*)(torch::Tensor, bool)>(&stochastic_round_bf16_cuda),
          "Stochastic rounding to BFloat16",
          py::arg("input"),
          py::arg("requires_grad") = false);
}
