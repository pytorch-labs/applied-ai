#include "stochastic_rounding.hpp"
#include <pybind11/pybind11.h>
namespace py = pybind11;


// Implementation of getOptimalBlockSize
__host__ int getOptimalBlockSize() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  return std::min(prop.maxThreadsPerBlock, 256);
}

// kernel wrapper
torch::Tensor stochastic_round_bf16_cuda(torch::Tensor input, bool requires_grad = false) {
  TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
  TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
  TORCH_CHECK(input.scalar_type() == torch::kFloat32,
              "Input tensor must be float32");

  const int threads_per_block = getOptimalBlockSize();
  const int num_elements = input.numel();

  // Calculate grid size
  const int max_blocks = 65535; // Maximum blocks per grid dimension
  const int min_elements_per_thread =
      4; // Each thread processes at least 4 elements
  const int target_blocks =
      (num_elements + min_elements_per_thread * threads_per_block - 1) /
      (min_elements_per_thread * threads_per_block);
  const int blocks = std::min(target_blocks, max_blocks);

  // Create output tensor - default is no grad for return but user selectable
  auto options = torch::TensorOptions()
                     .dtype(torch::kBFloat16)
                     .device(input.device())
                     .requires_grad(requires_grad);
  auto output = torch::empty_like(input, options);

  // Generate random seed
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<unsigned long long> dis;
  const unsigned long long seed = dis(gen);

  // Print debug info if failed to launch
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error before kernel launch: %s\n", cudaGetErrorString(err));
  }

  /*printf("Launching kernel with blocks=%d, threads_per_block=%d, "
         "num_elements=%d\n",
         blocks, threads_per_block, num_elements);
  */
  // Launch kernel
  stochastic_round_bf16<<<blocks, threads_per_block>>>(
      input.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16 *>(output.data_ptr()), num_elements, seed);

  // CUDA errors
  err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
              "CUDA kernel execution failed: ", cudaGetErrorString(err));

  // asynchronous errors
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
