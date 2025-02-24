#include "stochastic_rounding.hpp"
#include <random>

namespace py = pybind11;

__host__ int getOptimalBlockSize() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    return std::min(prop.maxThreadsPerBlock, 256);
}

template<typename T>
torch::Tensor launch_stochastic_round(
    torch::Tensor input,
    bool requires_grad,
    void (*kernel)(float*, T*, int, uint64_t),
    torch::ScalarType dtype) {

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
                      .dtype(dtype)
                      .device(input.device())
                      .requires_grad(requires_grad);
    auto output = torch::empty_like(input, options);

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dis;
    const uint64_t seed = dis(gen);

    kernel<<<num_blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        reinterpret_cast<T*>(output.data_ptr()),
        num_elements,
        seed);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "CUDA kernel execution failed: ", cudaGetErrorString(err));

    return output;
}

torch::Tensor stochastic_round_bf16_cuda(torch::Tensor input, bool requires_grad) {
    return launch_stochastic_round<__nv_bfloat16>(
        input, requires_grad, stochastic_round_bf16, torch::kBFloat16);
}

torch::Tensor stochastic_round_fp16_cuda(torch::Tensor input, bool requires_grad) {
    return launch_stochastic_round<__half>(
        input, requires_grad, stochastic_round_fp16, torch::kFloat16);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("stochastic_round_bf16",
          static_cast<torch::Tensor (*)(torch::Tensor, bool)>(&stochastic_round_bf16_cuda),
          "Stochastic rounding to BFloat16",
          py::arg("input"),
          py::arg("requires_grad") = false);

    m.def("stochastic_round_fp16",
          static_cast<torch::Tensor (*)(torch::Tensor, bool)>(&stochastic_round_fp16_cuda),
          "Stochastic rounding to Float16",
          py::arg("input"),
          py::arg("requires_grad") = false);
}
