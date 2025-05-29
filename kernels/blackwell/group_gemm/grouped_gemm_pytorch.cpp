#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <vector>

// Simple C interface - no CUTLASS headers
#include "grouped_gemm_kernel.h"

void validate_tensor(const torch::Tensor &tensor, const std::string &name,
                     torch::ScalarType expected_dtype) {
  TORCH_CHECK(tensor.is_cuda(), name + " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == expected_dtype,
              name + " must have dtype " + torch::toString(expected_dtype));
  TORCH_CHECK(tensor.is_contiguous(), name + " must be contiguous");
}

torch::Tensor grouped_gemm_func(const std::vector<torch::Tensor> &A_tensors,
                                const std::vector<torch::Tensor> &B_tensors,
                                const std::vector<torch::Tensor> &C_tensors,
                                float alpha = 1.0f, float beta = 0.0f,
                                bool use_2sm_config = false,
                                const std::string &raster_order = "M") {

  // Check requirements
  TORCH_CHECK(
      check_cuda_version_c() && check_gpu_architecture_c(),
      "This extension requires CUDA 12.8+ and Blackwell GPU (compute 10.0)");

  const int num_groups = A_tensors.size();
  TORCH_CHECK(num_groups > 0, "Must provide at least one group");
  TORCH_CHECK(B_tensors.size() == num_groups, "B_tensors size mismatch");
  TORCH_CHECK(C_tensors.size() == num_groups, "C_tensors size mismatch");
  TORCH_CHECK(raster_order == "M" || raster_order == "N",
              "raster_order must be 'M' or 'N'");

  c10::cuda::CUDAGuard device_guard(A_tensors[0].device());

  // Create C-compatible parameters
  GroupedGemmParams params;
  params.num_groups = num_groups;
  params.alpha = alpha;
  params.beta = beta;
  params.use_2sm_config = use_2sm_config;
  params.raster_order =
      (raster_order == "N") ? RASTER_ORDER_ALONG_N : RASTER_ORDER_ALONG_M;

  // Validate tensor types
  for (int i = 0; i < num_groups; ++i) {
    validate_tensor(A_tensors[i], "A_tensors[" + std::to_string(i) + "]",
                    torch::kFloat8_e4m3fn);
    validate_tensor(B_tensors[i], "B_tensors[" + std::to_string(i) + "]",
                    torch::kFloat8_e4m3fn);
    validate_tensor(C_tensors[i], "C_tensors[" + std::to_string(i) + "]",
                    torch::kFloat16);
  }

  // Allocate arrays for problem sizes and pointers
  std::vector<ProblemSize> problem_sizes(num_groups);
  std::vector<void *> A_ptrs(num_groups);
  std::vector<void *> B_ptrs(num_groups);
  std::vector<void *> C_ptrs(num_groups);

  // Fill in problem sizes and pointers
  for (int i = 0; i < num_groups; ++i) {
    // Get dimensions
    int M = A_tensors[i].size(0);
    int K = A_tensors[i].size(1);
    int N = B_tensors[i].size(1);

    // Validate dimensions
    TORCH_CHECK(B_tensors[i].size(0) == K, "Dimension mismatch: A[" +
                                               std::to_string(i) + "].K != B[" +
                                               std::to_string(i) + "].K");
    TORCH_CHECK(C_tensors[i].size(0) == M && C_tensors[i].size(1) == N,
                "C tensor dimensions don't match A @ B for group " +
                    std::to_string(i));

    // Set problem size
    problem_sizes[i].M = M;
    problem_sizes[i].N = N;
    problem_sizes[i].K = K;

    // Set pointers
    A_ptrs[i] = A_tensors[i].data_ptr();
    B_ptrs[i] = B_tensors[i].data_ptr();
    C_ptrs[i] = C_tensors[i].data_ptr();
  }

  // Set arrays in params
  params.problem_sizes = problem_sizes.data();
  params.A_ptrs = A_ptrs.data();
  params.B_ptrs = B_ptrs.data();
  params.C_ptrs = C_ptrs.data();

  // Run the grouped GEMM
  GroupedGemmResult result;
  if (use_2sm_config) {
    result = run_grouped_gemm_2sm_c(&params);
  } else {
    result = run_grouped_gemm_1sm_c(&params);
  }

  // Check for errors
  TORCH_CHECK(result.success,
              "Grouped GEMM failed: " + std::string(result.error_message));

  // Create a PyTorch tensor from the result
  auto options = torch::TensorOptions()
                     .dtype(torch::kFloat16)
                     .device(A_tensors[0].device());

  // Create a tensor that takes ownership of the output data
  torch::Tensor output = torch::from_blob(
      result.output_ptr, {static_cast<int64_t>(result.total_elements)},
      [result](void *ptr) { free_output_tensor_c(ptr); }, options);

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("grouped_gemm", &grouped_gemm_func, "Grouped GEMM for Blackwell GPUs",
        py::arg("A_tensors"), py::arg("B_tensors"), py::arg("C_tensors"),
        py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f,
        py::arg("use_2sm_config") = false, py::arg("raster_order") = "M");
}
