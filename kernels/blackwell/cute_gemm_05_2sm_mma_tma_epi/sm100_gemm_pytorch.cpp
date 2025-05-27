#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
torch::Tensor cute_blackwell_gemm_cuda(const torch::Tensor &A,
                                       const torch::Tensor &B,
                                       const torch::Tensor &C, float alpha,
                                       float beta);

torch::Tensor cute_blackwell_gemm_with_verification_cuda(
    const torch::Tensor &A, const torch::Tensor &B, const torch::Tensor &C,
    float alpha, float beta, bool verify, bool verbose);

// C++ interface
torch::Tensor cute_blackwell_gemm(const torch::Tensor &A,
                                  const torch::Tensor &B,
                                  const torch::Tensor &C, float alpha = 1.0f,
                                  float beta = 0.0f) {

  // Check that inputs are on GPU
  TORCH_CHECK(A.device().is_cuda(), "A must be on CUDA device");
  TORCH_CHECK(B.device().is_cuda(), "B must be on CUDA device");
  TORCH_CHECK(C.device().is_cuda(), "C must be on CUDA device");

  // Check that all tensors are on the same device
  TORCH_CHECK(A.device() == B.device(), "A and B must be on same device");
  TORCH_CHECK(A.device() == C.device(), "A and C must be on same device");

  return cute_blackwell_gemm_cuda(A, B, C, alpha, beta);
}

// C++ interface with verification
torch::Tensor cute_blackwell_gemm_verify(const torch::Tensor &A,
                                         const torch::Tensor &B,
                                         const torch::Tensor &C,
                                         float alpha = 1.0f, float beta = 0.0f,
                                         bool verify = true,
                                         bool verbose = true) {

  // Check that inputs are on GPU
  TORCH_CHECK(A.device().is_cuda(), "A must be on CUDA device");
  TORCH_CHECK(B.device().is_cuda(), "B must be on CUDA device");
  TORCH_CHECK(C.device().is_cuda(), "C must be on CUDA device");

  // Check that all tensors are on the same device
  TORCH_CHECK(A.device() == B.device(), "A and B must be on same device");
  TORCH_CHECK(A.device() == C.device(), "A and C must be on same device");

  return cute_blackwell_gemm_with_verification_cuda(A, B, C, alpha, beta,
                                                    verify, verbose);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cute_blackwell_gemm", &cute_blackwell_gemm, "CuTe Blackwell GEMM",
        py::arg("A"), py::arg("B"), py::arg("C"), py::arg("alpha") = 1.0f,
        py::arg("beta") = 0.0f);

  m.def("cute_blackwell_gemm_verify", &cute_blackwell_gemm_verify,
        "CuTe Blackwell GEMM with verification", py::arg("A"), py::arg("B"),
        py::arg("C"), py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f,
        py::arg("verify") = true, py::arg("verbose") = true);
}
