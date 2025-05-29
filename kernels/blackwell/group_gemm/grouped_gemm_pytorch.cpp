#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <iostream>
#include <torch/extension.h>
#include <vector>

// #include "grouped_gemm_kernel.h"

// CUDA error checking macro
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      throw std::runtime_error("CUDA error: " +                                \
                               std::string(cudaGetErrorString(error)));        \
    }                                                                          \
  } while (0)

// Convert PyTorch tensor to appropriate CUTLASS element type pointer
template <typename CutlassType>
CutlassType *get_cutlass_ptr(const torch::Tensor &tensor) {
  return reinterpret_cast<CutlassType *>(tensor.data_ptr());
}

// Validate tensor properties
void validate_tensor(const torch::Tensor &tensor, const std::string &name,
                     torch::ScalarType expected_dtype,
                     bool should_be_contiguous = true) {
  TORCH_CHECK(tensor.is_cuda(), name + " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == expected_dtype,
              name + " must have dtype " + torch::toString(expected_dtype));
  if (should_be_contiguous) {
    TORCH_CHECK(tensor.is_contiguous(), name + " must be contiguous");
  }
}

torch::Tensor grouped_gemm(const std::vector<torch::Tensor> &A_tensors,
                           const std::vector<torch::Tensor> &B_tensors,
                           const std::vector<torch::Tensor> &C_tensors,
                           float alpha = 1.0f, float beta = 0.0f,
                           bool use_2sm_config = false,
                           const std::string &raster_order = "M") {

  // Check CUDA version
  if (__CUDACC_VER_MAJOR__ < 12 ||
      (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 8)) {
    TORCH_CHECK(false, "This extension requires CUDA 12.8 or newer");
  }

  // Check GPU architecture
  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));
  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));

  if (!(props.major == 10 && props.minor == 0)) {
    TORCH_CHECK(false, "This extension requires NVIDIA Blackwell Architecture "
                       "(compute capability 10.0)");
  }

  const int num_groups = A_tensors.size();

  // Validate inputs
  TORCH_CHECK(num_groups > 0, "Must provide at least one group");
  TORCH_CHECK(B_tensors.size() == num_groups,
              "B_tensors must have same length as A_tensors");
  TORCH_CHECK(C_tensors.size() == num_groups,
              "C_tensors must have same length as A_tensors");
  TORCH_CHECK(raster_order == "M" || raster_order == "N",
              "raster_order must be 'M' or 'N'");

  c10::cuda::CUDAGuard device_guard(A_tensors[0].device());

  // Create parameters structure
  grouped_gemm::GroupedGemmParams params(num_groups);
  params.alpha = alpha;
  params.beta = beta;
  params.use_2sm_config = use_2sm_config;
  params.raster_order = (raster_order == "N")
                            ? grouped_gemm::RasterOrder::AlongN
                            : grouped_gemm::RasterOrder::AlongM;

  // Validate tensors and collect parameters
  for (int i = 0; i < num_groups; ++i) {
    const auto &A = A_tensors[i];
    const auto &B = B_tensors[i];
    const auto &C = C_tensors[i];

    // Validate tensor properties
    validate_tensor(A, "A[" + std::to_string(i) + "]", torch::kFloat8_e4m3fn);
    validate_tensor(B, "B[" + std::to_string(i) + "]", torch::kFloat8_e4m3fn);
    validate_tensor(C, "C[" + std::to_string(i) + "]", torch::kFloat16);

    TORCH_CHECK(A.dim() == 2, "A tensors must be 2D");
    TORCH_CHECK(B.dim() == 2, "B tensors must be 2D");
    TORCH_CHECK(C.dim() == 2, "C tensors must be 2D");

    int M = A.size(0);
    int K = A.size(1);
    int K_B = B.size(0);
    int N = B.size(1);
    int M_C = C.size(0);
    int N_C = C.size(1);

    TORCH_CHECK(K == K_B, "A.size(1) must equal B.size(0) for group " +
                              std::to_string(i));
    TORCH_CHECK(M == M_C && N == N_C,
                "C dimensions must match output dimensions for group " +
                    std::to_string(i));

    // Check for reasonable dimensions
    TORCH_CHECK(M > 0 && N > 0 && K > 0,
                "All dimensions must be positive for group " +
                    std::to_string(i));
    TORCH_CHECK(M <= 65536 && N <= 65536 && K <= 65536,
                "Dimensions too large for group " + std::to_string(i));

    // Add to parameters
    params.problem_sizes.emplace_back(M, N, K);
    params.A_ptrs.push_back(get_cutlass_ptr<ElementA>(A));
    params.B_ptrs.push_back(get_cutlass_ptr<ElementB>(B));
    params.C_ptrs.push_back(get_cutlass_ptr<ElementC>(C));
  }

  // Run the appropriate kernel
  grouped_gemm::GroupedGemmResult result;
  try {
    if (use_2sm_config) {
      result = grouped_gemm::run_grouped_gemm_2sm(params);
    } else {
      result = grouped_gemm::run_grouped_gemm_1sm(params);
    }
  } catch (const std::exception &e) {
    TORCH_CHECK(false,
                "Grouped GEMM execution failed: " + std::string(e.what()));
  }

  if (!result.success) {
    TORCH_CHECK(false, "Grouped GEMM execution failed");
  }

  // Create PyTorch tensor from result
  torch::Tensor output = torch::from_blob(
      result.output_ptr, {result.total_elements},
      [](void *ptr) {
        grouped_gemm::free_output_tensor(static_cast<ElementC *>(ptr));
      },
      torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));

  return output;
}

// Helper function to extract individual results from flattened output
std::vector<torch::Tensor>
grouped_gemm_split(const std::vector<torch::Tensor> &A_tensors,
                   const std::vector<torch::Tensor> &B_tensors,
                   const std::vector<torch::Tensor> &C_tensors,
                   float alpha = 1.0f, float beta = 0.0f,
                   bool use_2sm_config = false,
                   const std::string &raster_order = "M") {

  // Call the main function
  torch::Tensor flattened_result =
      grouped_gemm(A_tensors, B_tensors, C_tensors, alpha, beta, use_2sm_config,
                   raster_order);

  // Split the result back into individual tensors
  std::vector<torch::Tensor> results;
  results.reserve(A_tensors.size());

  int64_t offset = 0;
  for (size_t i = 0; i < A_tensors.size(); ++i) {
    int M = A_tensors[i].size(0);
    int N = B_tensors[i].size(1);
    int64_t num_elements = M * N;

    // Extract this group's result and reshape
    torch::Tensor group_result =
        flattened_result.slice(0, offset, offset + num_elements).view({M, N});
    results.push_back(group_result);

    offset += num_elements;
  }

  return results;
}

// Convenience function for single GEMM (compatibility)
torch::Tensor single_gemm(const torch::Tensor &A, const torch::Tensor &B,
                          const torch::Tensor &C, float alpha = 1.0f,
                          float beta = 0.0f, bool use_2sm_config = false,
                          const std::string &raster_order = "M") {

  std::vector<torch::Tensor> A_tensors = {A};
  std::vector<torch::Tensor> B_tensors = {B};
  std::vector<torch::Tensor> C_tensors = {C};

  torch::Tensor flattened_result =
      grouped_gemm(A_tensors, B_tensors, C_tensors, alpha, beta, use_2sm_config,
                   raster_order);

  // Reshape to match expected output dimensions
  int M = A.size(0);
  int N = B.size(1);
  return flattened_result.view({M, N});
}

// Function to check if the current GPU supports the extension
bool is_supported() {
  if (!torch::cuda::is_available()) {
    return false;
  }

  try {
    cudaDeviceProp props;
    int current_device_id;
    CUDA_CHECK(cudaGetDevice(&current_device_id));
    CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));

    // Check for Blackwell architecture
    if (props.major == 10 && props.minor == 0) {
      return true;
    }
  } catch (...) {
    return false;
  }

  return false;
}

// Function to get device information
std::string get_device_info() {
  if (!torch::cuda::is_available()) {
    return "CUDA not available";
  }

  try {
    cudaDeviceProp props;
    int current_device_id;
    CUDA_CHECK(cudaGetDevice(&current_device_id));
    CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));

    std::string info = "GPU: " + std::string(props.name) +
                       ", Compute: " + std::to_string(props.major) + "." +
                       std::to_string(props.minor) +
                       ", SMs: " + std::to_string(props.multiProcessorCount);

    return info;
  } catch (const std::exception &e) {
    return "Error getting device info: " + std::string(e.what());
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Grouped GEMM using CUTLASS for NVIDIA Blackwell Architecture";

  m.def("grouped_gemm", &grouped_gemm,
        "Perform grouped GEMM operations: D = alpha * A @ B + beta * C",
        py::arg("A_tensors"), py::arg("B_tensors"), py::arg("C_tensors"),
        py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f,
        py::arg("use_2sm_config") = false, py::arg("raster_order") = "M");

  m.def("grouped_gemm_split", &grouped_gemm_split,
        "Perform grouped GEMM and return individual result tensors",
        py::arg("A_tensors"), py::arg("B_tensors"), py::arg("C_tensors"),
        py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f,
        py::arg("use_2sm_config") = false, py::arg("raster_order") = "M");

  m.def("single_gemm", &single_gemm,
        "Perform single GEMM operation: D = alpha * A @ B + beta * C",
        py::arg("A"), py::arg("B"), py::arg("C"), py::arg("alpha") = 1.0f,
        py::arg("beta") = 0.0f, py::arg("use_2sm_config") = false,
        py::arg("raster_order") = "M");

  m.def("is_supported", &is_supported,
        "Check if the current GPU supports this extension");

  m.def("get_device_info", &get_device_info,
        "Get information about the current GPU device");
}
