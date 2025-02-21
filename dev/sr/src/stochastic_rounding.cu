 #include "stochastic_rounding.hpp"
#include <pybind11/pybind11.h>
namespace py = pybind11;

// Define block size if not already defined
#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 256
#endif

#ifndef ELEMENTS_PER_THREAD
#define ELEMENTS_PER_THREAD 8
#endif

#ifndef VECTOR_SIZE
#define VECTOR_SIZE 4
#endif

__global__ void stochastic_round_bf16(const float4* __restrict__ input,
                                     __nv_bfloat16* __restrict__ output,
                                     const int size,
                                     const unsigned long long seed);

// Implementation of getOptimalBlockSize
__host__ int getOptimalBlockSize() {
    return THREADS_PER_BLOCK;
}

// kernel wrapper
torch::Tensor stochastic_round_bf16_cuda(torch::Tensor input, bool requires_grad) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32,
                "Input tensor must be float32");

    const int num_elements = input.numel();
    const int num_elements4 = (num_elements + 3) / 4;

    // Calculate grid size based on vector processing
    const int vector_elements = VECTOR_SIZE * ELEMENTS_PER_THREAD;
    const int elements_per_block = THREADS_PER_BLOCK * vector_elements;
    const int num_blocks = (num_elements + elements_per_block - 1) / elements_per_block;
    const int max_blocks = 128000; // Increased for Hopper
    const int blocks = std::min(num_blocks, max_blocks);

    // Create output tensor with proper options
    auto options = torch::TensorOptions()
                      .dtype(torch::kBFloat16)
                      .device(input.device())
                      .requires_grad(requires_grad)
                      .memory_format(torch::MemoryFormat::Contiguous);
    auto output = torch::empty_like(input, options);

    // Generate random seed with high entropy
    std::random_device rd;
    std::seed_seq seq{rd(), rd(), rd(), rd()};
    std::mt19937_64 gen(seq);
    std::uniform_int_distribution<unsigned long long> dis;
    const unsigned long long seed = dis(gen);

    // Set L1 cache preference
    //cudaFuncSetCacheConfig(stochastic_round_bf16, cudaFuncCachePreferL1);

    // Print debug info if in debug mode
    #ifdef DEBUG
    printf("Launching kernel with blocks=%d, threads_per_block=%d, "
           "num_elements=%d, vector_size=%d, elements_per_thread=%d\n",
           blocks, THREADS_PER_BLOCK, num_elements, VECTOR_SIZE, ELEMENTS_PER_THREAD);
    #endif

    // Check for errors before launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error before kernel launch: %s\n", cudaGetErrorString(err));
    }

    // Ensure alignment for float4
    TORCH_CHECK((reinterpret_cast<std::uintptr_t>(input.data_ptr<float>()) % (sizeof(float4))) == 0,
                "Input tensor must be aligned to float4 boundary");

    // Launch kernel with proper pointer alignment
    float* input_ptr = input.data_ptr<float>();
    stochastic_round_bf16<<<blocks, THREADS_PER_BLOCK>>>(
        reinterpret_cast<const float4*>(input_ptr),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        num_elements,
        seed
    );

    // Check for kernel launch errors
    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "CUDA kernel execution failed: ", cudaGetErrorString(err));

    // Check for asynchronous errors
    err = cudaDeviceSynchronize();
    TORCH_CHECK(err == cudaSuccess,
                "CUDA synchronization failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("stochastic_round_bf16",
          static_cast<torch::Tensor (*)(torch::Tensor, bool)>(&stochastic_round_bf16_cuda),
          "Stochastic rounding to BFloat16",
          py::arg("input"),
          py::arg("requires_grad") = false);
}
