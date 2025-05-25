// sm100_gemm_pytorch.cpp - PyTorch C++ extension (no CUDA code)
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "sm100_gemm.h"

// Check if SM100 support is available at compile time
bool is_sm100_supported() {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  return true;
#else
  return false;
#endif
}

// Check if current GPU supports SM100 at runtime
bool check_sm100_device() {
  int device;
  cudaGetDevice(&device);

  cudaDeviceProp props;
  cudaError_t error = cudaGetDeviceProperties(&props, device);
  if (error != cudaSuccess) {
    return false;
  }

  // Check for SM100 architecture (compute capability 10.0a)
  return (props.major == 10 && props.minor == 0);
}

torch::Tensor sm100_gemm_f16(const torch::Tensor &A, const torch::Tensor &B,
                             const torch::Tensor &C, float alpha = 1.0f,
                             float beta = 0.0f) {

  // Check compile-time support
  TORCH_CHECK(
      is_sm100_supported(),
      "SM100 support not compiled. Requires CUTLASS_ARCH_MMA_SM100_SUPPORTED");

  // Check runtime device support
  TORCH_CHECK(check_sm100_device(),
              "Current GPU does not support SM100 architecture (requires "
              "compute capability 10.0a)");

  // Input validation
  TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
  TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
  TORCH_CHECK(C.device().is_cuda(), "C must be a CUDA tensor");
  TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
  TORCH_CHECK(B.dtype() == torch::kFloat16, "B must be float16");
  TORCH_CHECK(C.dtype() == torch::kFloat32, "C must be float32");
  TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
  TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
  TORCH_CHECK(C.is_contiguous(), "C must be contiguous");
  TORCH_CHECK(A.dim() == 2, "A must be 2D");
  TORCH_CHECK(B.dim() == 2, "B must be 2D");
  TORCH_CHECK(C.dim() == 2, "C must be 2D");

  // Get dimensions
  int64_t M = A.size(0);
  int64_t K = A.size(1);
  int64_t N = B.size(0);
  int64_t K_B = B.size(1);

  TORCH_CHECK(K == K_B, "Inner dimensions must match: A.shape[1]=", K,
              ", B.shape[1]=", K_B);
  TORCH_CHECK(C.size(0) == M && C.size(1) == N, "C dimensions (", C.size(0),
              ", ", C.size(1), ") must match output shape (", M, ", ", N, ")");

  // Check alignment requirements for SM100
  TORCH_CHECK(M % 128 == 0, "M=", M, " must be multiple of 128");
  TORCH_CHECK(N % 256 == 0, "N=", N, " must be multiple of 256");
  TORCH_CHECK(K % 64 == 0, "K=", K, " must be multiple of 64");

  // Check size limits (avoid overflow in int conversion)
  TORCH_CHECK(M <= INT_MAX && N <= INT_MAX && K <= INT_MAX,
              "Dimensions too large for int conversion");

  // Create output tensor
  auto D = torch::empty_like(C);

  // Set CUDA device guard
  const auto device = A.device();
  c10::cuda::CUDAGuard device_guard(device);

  // Get current CUDA stream
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(device.index()).stream();

  // Launch the kernel
  cudaError_t error = launch_sm100_gemm_f16_tma(
      A.data_ptr(), B.data_ptr(), C.data_ptr(), D.data_ptr(),
      static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), alpha,
      beta, stream);

  // Check for launch errors
  TORCH_CHECK(error == cudaSuccess,
              "SM100 GEMM kernel launch failed: ", cudaGetErrorString(error));

  // Check for kernel execution errors
  C10_CUDA_CHECK(cudaGetLastError());

  return D;
}

// Utility functions for debugging and information
torch::Tensor get_device_info() {
  int device;
  cudaGetDevice(&device);

  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, device);

  // Return device info as a tensor (for easy Python access)
  auto info = torch::zeros({4}, torch::kInt32);
  auto accessor = info.accessor<int32_t, 1>();

  accessor[0] = props.major;          // Compute capability major
  accessor[1] = props.minor;          // Compute capability minor
  accessor[2] = is_sm100_supported(); // Compile-time support
  accessor[3] = check_sm100_device(); // Runtime device support

  return info;
}

std::vector<int64_t> get_aligned_shape(int64_t M, int64_t N, int64_t K) {
  // Return properly aligned dimensions for SM100
  int64_t aligned_M = ((M + 127) / 128) * 128;
  int64_t aligned_N = ((N + 255) / 256) * 256;
  int64_t aligned_K = ((K + 63) / 64) * 64;

  return {aligned_M, aligned_N, aligned_K};
}

torch::Tensor create_aligned_tensor(const std::vector<int64_t> &shape,
                                    torch::ScalarType dtype,
                                    torch::Device device) {
  // Create a tensor with SM100-aligned dimensions
  TORCH_CHECK(shape.size() == 2, "Shape must be 2D");

  auto aligned_shape =
      get_aligned_shape(shape[0], shape[1], shape.size() > 2 ? shape[2] : 64);

  if (shape.size() == 2) {
    return torch::zeros({aligned_shape[0], aligned_shape[1]},
                        torch::TensorOptions().dtype(dtype).device(device));
  } else {
    return torch::zeros({aligned_shape[0], aligned_shape[2]},
                        torch::TensorOptions().dtype(dtype).device(device));
  }
}

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "SM100 GEMM PyTorch Extension";

  // Main GEMM function
  m.def("sm100_gemm_f16", &sm100_gemm_f16,
        "SM100 GEMM with FP16 inputs and FP32 output: D = alpha * A @ B^T + "
        "beta * C",
        py::arg("A"), py::arg("B"), py::arg("C"), py::arg("alpha") = 1.0f,
        py::arg("beta") = 0.0f);

  // Utility functions
  m.def("is_sm100_supported", &is_sm100_supported,
        "Check if SM100 support was compiled in");

  m.def("check_sm100_device", &check_sm100_device,
        "Check if current GPU supports SM100 architecture");

  m.def("get_device_info", &get_device_info,
        "Get device compute capability and SM100 support info");

  m.def("get_aligned_shape", &get_aligned_shape,
        "Get SM100-aligned dimensions for given shape", py::arg("M"),
        py::arg("N"), py::arg("K"));

  m.def("create_aligned_tensor", &create_aligned_tensor,
        "Create tensor with SM100-aligned dimensions", py::arg("shape"),
        py::arg("dtype"), py::arg("device"));

  // Constants for alignment requirements
  m.attr("MMA_TILE_M") = 128;
  m.attr("MMA_TILE_N") = 256;
  m.attr("MMA_TILE_K") = 64;
}
