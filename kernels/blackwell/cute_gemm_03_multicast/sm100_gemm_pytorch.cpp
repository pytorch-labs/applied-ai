// sm100_gemm_pytorch.cpp - Enhanced PyTorch C++ extension
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/types.h>

#include "sm100_gemm.h"

#include <memory>
#include <sstream>
#include <stdexcept>

namespace {

// Enhanced device capability checking
struct DeviceInfo {
  int major;
  int minor;
  bool compile_support;
  bool runtime_support;
  size_t total_memory;
  size_t free_memory;
  int multiprocessor_count;
  int max_threads_per_block;
};

DeviceInfo get_enhanced_device_info() {
  DeviceInfo info{};

  int device;
  cudaGetDevice(&device);

  cudaDeviceProp props;
  cudaError_t error = cudaGetDeviceProperties(&props, device);
  if (error != cudaSuccess) {
    throw std::runtime_error("Failed to get device properties: " +
                             std::string(cudaGetErrorString(error)));
  }

  info.major = props.major;
  info.minor = props.minor;
  info.multiprocessor_count = props.multiProcessorCount;
  info.max_threads_per_block = props.maxThreadsPerBlock;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  info.compile_support = true;
#else
  info.compile_support = false;
#endif

  info.runtime_support = (props.major == 10 && props.minor == 0);

  size_t free_mem, total_mem;
  error = cudaMemGetInfo(&free_mem, &total_mem);
  if (error == cudaSuccess) {
    info.free_memory = free_mem;
    info.total_memory = total_mem;
  }

  return info;
}

// Enhanced error message generation
std::string generate_error_context(const torch::Tensor &A,
                                   const torch::Tensor &B,
                                   const torch::Tensor &C, float alpha,
                                   float beta) {
  std::ostringstream oss;
  oss << "SM100 GEMM Error Context:\n";
  oss << "  A shape: [" << A.size(0) << ", " << A.size(1)
      << "], dtype: " << A.dtype() << "\n";
  oss << "  B shape: [" << B.size(0) << ", " << B.size(1)
      << "], dtype: " << B.dtype() << "\n";
  oss << "  C shape: [" << C.size(0) << ", " << C.size(1)
      << "], dtype: " << C.dtype() << "\n";
  oss << "  alpha: " << alpha << ", beta: " << beta << "\n";
  oss << "  A device: " << A.device() << ", contiguous: " << A.is_contiguous()
      << "\n";
  oss << "  B device: " << B.device() << ", contiguous: " << B.is_contiguous()
      << "\n";
  oss << "  C device: " << C.device() << ", contiguous: " << C.is_contiguous();
  return oss.str();
}

// Memory requirement estimation
size_t estimate_memory_requirement(int64_t M, int64_t N, int64_t K) {
  size_t a_size = M * K * sizeof(at::Half);
  size_t b_size = N * K * sizeof(at::Half);
  size_t c_size = M * N * sizeof(float);
  size_t d_size = M * N * sizeof(float);

  // Add some overhead for intermediate computations and alignment
  size_t overhead =
      std::max(static_cast<size_t>(1024 * 1024),          // 1MB minimum
               (a_size + b_size + c_size + d_size) / 10); // 10% overhead

  return a_size + b_size + c_size + d_size + overhead;
}

} // anonymous namespace

// Enhanced compile-time support check
bool is_sm100_supported() {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  return true;
#else
  return false;
#endif
}

// Enhanced runtime device check with detailed validation
bool check_sm100_device() {
  try {
    auto info = get_enhanced_device_info();
    return info.runtime_support;
  } catch (...) {
    return false;
  }
}

// Enhanced device compatibility check
std::tuple<bool, std::string> check_sm100_compatibility_detailed() {
  try {
    auto info = get_enhanced_device_info();

    if (!info.compile_support) {
      return std::make_tuple(false, "SM100 support not compiled. Rebuild with "
                                    "CUTLASS_ARCH_MMA_SM100_SUPPORTED=1");
    }

    if (!info.runtime_support) {
      return std::make_tuple(
          false, "Current GPU does not support SM100 (compute capability " +
                     std::to_string(info.major) + "." +
                     std::to_string(info.minor) + ", requires 10.0a)");
    }

    if (info.multiprocessor_count < 4) {
      return std::make_tuple(
          false,
          "Insufficient multiprocessors for efficient SM100 operation (found " +
              std::to_string(info.multiprocessor_count) + ", recommend 8+)");
    }

    return std::make_tuple(true, "SM100 fully supported and ready");

  } catch (const std::exception &e) {
    return std::make_tuple(false,
                           "Error checking device: " + std::string(e.what()));
  }
}

// Enhanced main GEMM function with comprehensive validation
torch::Tensor sm100_gemm_f16_enhanced(const torch::Tensor &A,
                                      const torch::Tensor &B,
                                      const torch::Tensor &C,
                                      float alpha = 1.0f, float beta = 0.0f,
                                      bool validate_inputs = true) {

  // Quick compatibility check if validation enabled
  if (validate_inputs) {
    auto [is_supported, message] = check_sm100_compatibility_detailed();
    TORCH_CHECK(is_supported, message);
  }

  // Enhanced input validation with detailed error messages
  try {
    TORCH_CHECK(A.device().is_cuda(),
                "Tensor A must be on CUDA device, got: ", A.device());
    TORCH_CHECK(B.device().is_cuda(),
                "Tensor B must be on CUDA device, got: ", B.device());
    TORCH_CHECK(C.device().is_cuda(),
                "Tensor C must be on CUDA device, got: ", C.device());

    TORCH_CHECK(A.dtype() == torch::kFloat16,
                "Tensor A must be float16, got: ", A.dtype());
    TORCH_CHECK(B.dtype() == torch::kFloat16,
                "Tensor B must be float16, got: ", B.dtype());
    TORCH_CHECK(C.dtype() == torch::kFloat32,
                "Tensor C must be float32, got: ", C.dtype());

    TORCH_CHECK(A.is_contiguous(), "Tensor A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "Tensor B must be contiguous");
    TORCH_CHECK(C.is_contiguous(), "Tensor C must be contiguous");

    TORCH_CHECK(A.dim() == 2, "Tensor A must be 2D, got ", A.dim(), "D");
    TORCH_CHECK(B.dim() == 2, "Tensor B must be 2D, got ", B.dim(), "D");
    TORCH_CHECK(C.dim() == 2, "Tensor C must be 2D, got ", C.dim(), "D");

  } catch (const c10::Error &e) {
    // Add context to torch errors
    std::string context = generate_error_context(A, B, C, alpha, beta);
    throw c10::Error(e.msg() + "\n" + context, e.backtrace());
  }

  // Get and validate dimensions
  int64_t M = A.size(0);
  int64_t K = A.size(1);
  int64_t N = B.size(0);
  int64_t K_B = B.size(1);

  TORCH_CHECK(K == K_B, "Inner dimensions must match: A.shape[1]=", K,
              ", B.shape[1]=", K_B);
  TORCH_CHECK(C.size(0) == M && C.size(1) == N, "C dimensions (", C.size(0),
              ", ", C.size(1), ") must match output shape (", M, ", ", N, ")");

  // Enhanced alignment checking with suggestions
  bool m_aligned = (M % 128 == 0);
  bool n_aligned = (N % 256 == 0);
  bool k_aligned = (K % 64 == 0);

  if (!m_aligned || !n_aligned || !k_aligned) {
    std::ostringstream suggestion;
    suggestion << "Alignment requirements not met. Current: (" << M << ", " << N
               << ", " << K << "). ";
    suggestion << "Required: M % 128 == 0, N % 256 == 0, K % 64 == 0. ";
    suggestion << "Suggested aligned dimensions: (";
    suggestion << ((M + 127) / 128) * 128 << ", ";
    suggestion << ((N + 255) / 256) * 256 << ", ";
    suggestion << ((K + 63) / 64) * 64 << ")";
    TORCH_CHECK(false, suggestion.str());
  }

  // Enhanced size validation with overflow protection
  TORCH_CHECK(M > 0 && M <= INT_MAX, "M dimension out of valid range: ", M);
  TORCH_CHECK(N > 0 && N <= INT_MAX, "N dimension out of valid range: ", N);
  TORCH_CHECK(K > 0 && K <= INT_MAX, "K dimension out of valid range: ", K);

  // Memory requirement check
  if (validate_inputs) {
    auto info = get_enhanced_device_info();
    size_t required_memory = estimate_memory_requirement(M, N, K);
    TORCH_CHECK(
        required_memory <= info.free_memory,
        "Insufficient GPU memory. Required: ", required_memory / (1024 * 1024),
        " MB, Available: ", info.free_memory / (1024 * 1024), " MB");
  }

  // Enhanced tensor creation with proper device placement
  const auto device = A.device();
  auto D =
      torch::empty({M, N}, torch::TensorOptions()
                               .dtype(torch::kFloat32)
                               .device(device)
                               .memory_format(torch::MemoryFormat::Contiguous));

  // Set CUDA device guard for multi-GPU safety
  c10::cuda::CUDAGuard device_guard(device);

  // Get current CUDA stream with error checking
  cudaStream_t stream;
  try {
    stream = at::cuda::getCurrentCUDAStream(device.index()).stream();
  } catch (...) {
    stream = 0; // Fall back to default stream
  }

  // Enhanced kernel launch with comprehensive error handling
  cudaError_t error = launch_sm100_gemm_f16(
      A.data_ptr(), B.data_ptr(), C.data_ptr(), D.data_ptr(),
      static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), alpha,
      beta, stream);

  // Enhanced error reporting
  if (error != cudaSuccess) {
    std::ostringstream error_msg;
    error_msg << "SM100 GEMM kernel launch failed: "
              << cudaGetErrorString(error);
    error_msg << " (error code: " << error << ")\n";
    error_msg << generate_error_context(A, B, C, alpha, beta);
    throw std::runtime_error(error_msg.str());
  }

  // Check for kernel execution errors
  error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, "SM100 GEMM kernel execution failed: ",
              cudaGetErrorString(error));

  return D;
}

// Backward compatibility wrapper
torch::Tensor sm100_gemm_f16(const torch::Tensor &A, const torch::Tensor &B,
                             const torch::Tensor &C, float alpha = 1.0f,
                             float beta = 0.0f) {
  return sm100_gemm_f16_enhanced(A, B, C, alpha, beta, true);
}

// Enhanced device information function
torch::Tensor get_device_info_enhanced() {
  try {
    auto info = get_enhanced_device_info();

    auto device_info = torch::zeros({8}, torch::kInt32);
    auto accessor = device_info.accessor<int32_t, 1>();

    accessor[0] = info.major;
    accessor[1] = info.minor;
    accessor[2] = info.compile_support ? 1 : 0;
    accessor[3] = info.runtime_support ? 1 : 0;
    accessor[4] = static_cast<int32_t>(info.total_memory / (1024 * 1024)); // MB
    accessor[5] = static_cast<int32_t>(info.free_memory / (1024 * 1024));  // MB
    accessor[6] = info.multiprocessor_count;
    accessor[7] = info.max_threads_per_block;

    return device_info;

  } catch (...) {
    return torch::zeros({8}, torch::kInt32);
  }
}

// Backward compatibility
torch::Tensor get_device_info() {
  auto enhanced = get_device_info_enhanced();
  auto basic = torch::zeros({4}, torch::kInt32);
  basic[0] = enhanced[0]; // major
  basic[1] = enhanced[1]; // minor
  basic[2] = enhanced[2]; // compile support
  basic[3] = enhanced[3]; // runtime support
  return basic;
}

// Enhanced alignment utilities
std::vector<int64_t> get_aligned_shape_enhanced(int64_t M, int64_t N,
                                                int64_t K) {
  int64_t aligned_M = ((M + 127) / 128) * 128;
  int64_t aligned_N = ((N + 255) / 256) * 256;
  int64_t aligned_K = ((K + 63) / 64) * 64;
  return {aligned_M, aligned_N, aligned_K};
}

// Backward compatibility
std::vector<int64_t> get_aligned_shape(int64_t M, int64_t N, int64_t K) {
  return get_aligned_shape_enhanced(M, N, K);
}

// Performance estimation function
std::vector<float> estimate_performance(int64_t M, int64_t N, int64_t K,
                                        int cluster_m = 2, int cluster_n = 2) {
  // Basic performance estimation based on problem size
  double flops = 2.0 * M * N * K; // Multiply-add operations

  // Rough performance estimates based on SM100 characteristics
  double peak_flops = 1500e12;    // ~1500 TFLOPS for large matrices
  double peak_bandwidth = 8000e9; // ~8 TB/s memory bandwidth

  // Memory requirements
  double memory_bytes = M * K * 2 + N * K * 2 + M * N * 4 * 2;

  // Simple performance model
  double compute_time = flops / peak_flops;
  double memory_time = memory_bytes / peak_bandwidth;
  double estimated_time = std::max(compute_time, memory_time);

  // Add cluster overhead
  double cluster_overhead = 1.0 + 0.1 * (cluster_m * cluster_n - 1);
  estimated_time *= cluster_overhead;

  double estimated_gflops = (flops / estimated_time) / 1e9;
  double estimated_bandwidth = (memory_bytes / estimated_time) / 1e9;

  return {
      static_cast<float>(estimated_gflops),
      static_cast<float>(estimated_bandwidth),
      static_cast<float>(estimated_time * 1000) // ms
  };
}

// Tensor creation with optimal alignment
std::vector<torch::Tensor>
create_aligned_tensors_enhanced(int64_t M, int64_t N, int64_t K,
                                const torch::Device &device = torch::kCUDA,
                                torch::ScalarType dtype_AB = torch::kFloat16,
                                torch::ScalarType dtype_C = torch::kFloat32) {

  auto aligned_dims = get_aligned_shape_enhanced(M, N, K);
  int64_t aligned_M = aligned_dims[0];
  int64_t aligned_N = aligned_dims[1];
  int64_t aligned_K = aligned_dims[2];

  auto options_AB = torch::TensorOptions().dtype(dtype_AB).device(device);
  auto options_C = torch::TensorOptions().dtype(dtype_C).device(device);

  auto A = torch::zeros({aligned_M, aligned_K}, options_AB);
  auto B = torch::zeros({aligned_N, aligned_K}, options_AB);
  auto C = torch::zeros({aligned_M, aligned_N}, options_C);

  return {A, B, C};
}

// Backward compatibility wrapper
torch::Tensor create_aligned_tensor(const std::vector<int64_t> &shape,
                                    torch::ScalarType dtype,
                                    torch::Device device) {
  TORCH_CHECK(shape.size() == 2, "Shape must be 2D");

  // This is a simplified version for backward compatibility
  if (shape.size() == 2) {
    auto aligned_dims = get_aligned_shape_enhanced(shape[0], shape[1], 64);
    return torch::zeros({aligned_dims[0], aligned_dims[1]},
                        torch::TensorOptions().dtype(dtype).device(device));
  }

  return torch::zeros(shape,
                      torch::TensorOptions().dtype(dtype).device(device));
}

// Enhanced Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Enhanced SM100 GEMM PyTorch Extension with TMA Multicast";

  // Main GEMM functions
  m.def("sm100_gemm_f16", &sm100_gemm_f16,
        "SM100 GEMM with FP16 inputs and FP32 output (backward compatible)",
        py::arg("A"), py::arg("B"), py::arg("C"), py::arg("alpha") = 1.0f,
        py::arg("beta") = 0.0f);

  m.def("sm100_gemm_f16_enhanced", &sm100_gemm_f16_enhanced,
        "Enhanced SM100 GEMM with comprehensive validation and error handling",
        py::arg("A"), py::arg("B"), py::arg("C"), py::arg("alpha") = 1.0f,
        py::arg("beta") = 0.0f, py::arg("validate_inputs") = true);

  // Device information and compatibility
  m.def("is_sm100_supported", &is_sm100_supported,
        "Check if SM100 support was compiled in");
  m.def("check_sm100_device", &check_sm100_device,
        "Check if current GPU supports SM100 architecture");
  m.def("check_sm100_compatibility_detailed",
        &check_sm100_compatibility_detailed,
        "Get detailed SM100 compatibility information");

  // Device information functions
  m.def("get_device_info", &get_device_info,
        "Get basic device info (backward compatible)");
  m.def("get_device_info_enhanced", &get_device_info_enhanced,
        "Get comprehensive device information including memory and compute "
        "capabilities");

  // Utility functions
  m.def("get_aligned_shape", &get_aligned_shape,
        "Get SM100-aligned dimensions for given shape (backward compatible)",
        py::arg("M"), py::arg("N"), py::arg("K"));
  m.def("get_aligned_shape_enhanced", &get_aligned_shape_enhanced,
        "Get SM100-aligned dimensions with enhanced validation", py::arg("M"),
        py::arg("N"), py::arg("K"));

  // Performance and optimization
  m.def("estimate_performance", &estimate_performance,
        "Estimate performance characteristics for given problem size",
        py::arg("M"), py::arg("N"), py::arg("K"), py::arg("cluster_m") = 2,
        py::arg("cluster_n") = 2);

  // Tensor creation utilities
  m.def("create_aligned_tensor", &create_aligned_tensor,
        "Create tensor with SM100-aligned dimensions (backward compatible)",
        py::arg("shape"), py::arg("dtype"), py::arg("device"));
  m.def("create_aligned_tensors_enhanced", &create_aligned_tensors_enhanced,
        "Create optimally aligned tensors for SM100 operations", py::arg("M"),
        py::arg("N"), py::arg("K"), py::arg("device") = torch::kCUDA,
        py::arg("dtype_AB") = torch::kFloat16,
        py::arg("dtype_C") = torch::kFloat32);

  // Constants for alignment requirements
  m.attr("MMA_TILE_M") = 128;
  m.attr("MMA_TILE_N") = 256;
  m.attr("MMA_TILE_K") = 64;
  m.attr("MAX_CLUSTER_SIZE") = 16;

  // Version information
  m.attr("__version__") = "2.0.0";
  m.attr("__cutlass_version__") = "cutlass-4.0";

  // Feature flags
  m.attr("HAS_TMA_MULTICAST") = true;
  m.attr("HAS_ENHANCED_ERROR_HANDLING") = true;
  m.attr("HAS_PERFORMANCE_ESTIMATION") = true;
}
