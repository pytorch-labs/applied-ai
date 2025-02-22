#include "stochastic_rounding.hpp"
#include <pybind11/pybind11.h>
namespace py = pybind11;

torch::Tensor stochastic_round_bf16_cuda(torch::Tensor input, bool requires_grad) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32,
                "Input tensor must be float32");

    const int num_elements = input.numel();
    // Pad input size to multiple of 4 for vectorized access
    const int padded_elements = ((num_elements + 3) / 4) * 4;
    const int vec4_elements = padded_elements / 4;

    const int threads_per_block = 256;  // Optimal for Hopper

    // Get device properties for optimal launch configuration
    int num_sms, max_threads_per_sm;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, input.device().index());
    cudaDeviceGetAttribute(&max_threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, input.device().index());

    const int max_blocks_per_sm = max_threads_per_sm / threads_per_block;
    const int min_blocks_per_sm = 4;  // Target occupancy
    const int max_blocks = num_sms * std::min(min_blocks_per_sm, max_blocks_per_sm);

    const int target_blocks = (vec4_elements + threads_per_block - 1) / threads_per_block;
    const int num_blocks = std::min(target_blocks, max_blocks);

    // Create output tensor
    auto options = torch::TensorOptions()
                      .dtype(torch::kBFloat16)
                      .device(input.device())
                      .requires_grad(requires_grad);
    auto output = torch::empty({padded_elements}, options);

    // Generate random seed
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<unsigned long long> dis;
    const unsigned long long seed = dis(gen);

    // Launch kernel
    stochastic_round_bf16<<<num_blocks, threads_per_block>>>(
        reinterpret_cast<const float4*>(input.data_ptr<float>()),
        reinterpret_cast<__nv_bfloat162*>(output.data_ptr()),
        vec4_elements,
        seed
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "CUDA kernel execution failed: ", cudaGetErrorString(err));

    // Return correct size tensor
    if (padded_elements > num_elements) {
        return output.slice(0, 0, num_elements);
    }
    return output;
}

// Python binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("stochastic_round_bf16",
          static_cast<torch::Tensor (*)(torch::Tensor, bool)>(&stochastic_round_bf16_cuda),
          "Stochastic rounding to BFloat16",
          py::arg("input"),
          py::arg("requires_grad") = false);
}
