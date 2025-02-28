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

    // Calc grid launch
    const int threads_per_block = 256;  // Fixed size for better occupancy
    const int num_elements = input.numel();
    const int elements_per_thread = 4;  // Vector size (float4)

    // Calculate blocks based only on data size (rounded up)
    const int data_blocks = (num_elements + elements_per_thread * threads_per_block - 1) /
                            (elements_per_thread * threads_per_block);

    // Ensure minimum number of blocks for good utilization on any GPU
    // This is a fixed number that works well across different GPU architectures
     // Should work well across different GPU generations

    // Calculate optimal block count based on input size
    //const int min_blocks = (num_elements > 1009000000) ? 2048 : 32;
    //const int num_blocks = std::max(data_blocks, min_blocks);


    // Dynamic launch strategy based on num elements size
    int num_blocks;
    if (num_elements < 1050000001) {  // Threshold on Hopper
        // For small datasets, ensure minimum blocks for good GPU utilization
        const int min_blocks = 32;
        num_blocks = std::max(data_blocks, min_blocks);
        // Print block launch info
        //printf("Launching kernel with blocks=%d, threads_per_block=%d, num_elements=%d\n",
        //    num_blocks, threads_per_block, num_elements);
    }
    else  {  // Huge tensor (1B+ elements)...max we can go w/o IMA
        num_blocks = 135168;
    }



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
