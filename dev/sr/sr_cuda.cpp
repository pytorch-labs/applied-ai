#include <torch/extension.h>
#include <vector>

// Declaration of CUDA kernel launcher
torch::Tensor stochastic_round_cuda_forward(torch::Tensor input);

// Python-visible function
torch::Tensor stochastic_round_forward(torch::Tensor input) {

  TORCH_CHECK(input.is_cuda(), "Input tensor must be on GPU");
  TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");

  return stochastic_round_cuda_forward(input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &stochastic_round_forward, "Stochastic Round Forward");
}
