#include <torch/extension.h>

// Forward declare the CUDA function
torch::Tensor blackwell_gemm_f16(const torch::Tensor &A, const torch::Tensor &B,
                                 const torch::Tensor &C, float alpha,
                                 float beta);

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Blackwell SM100 GEMM PyTorch Extension";

  m.def("gemm_f16", &blackwell_gemm_f16,
        "Blackwell SM100 F16xF16->F32 GEMM with tcgen05.mma and TMA",
        py::arg("A"), py::arg("B"), py::arg("C"), py::arg("alpha") = 1.0f,
        py::arg("beta") = 0.0f);
}
