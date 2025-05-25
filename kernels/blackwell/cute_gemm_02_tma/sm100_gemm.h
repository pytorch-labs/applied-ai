// sm100_gemm_kernel.h - Header file for CUDA kernel
#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Launch SM100 GEMM kernel: D = alpha * A @ B^T + beta * C
 *
 * @param d_A Pointer to matrix A in device memory (M x K, FP16, K-major)
 * @param d_B Pointer to matrix B in device memory (N x K, FP16, K-major)
 * @param d_C Pointer to matrix C in device memory (M x N, FP32, N-major)
 * @param d_D Pointer to matrix D in device memory (M x N, FP32, N-major)
 * @param M Number of rows in A and C/D
 * @param N Number of rows in B and columns in C/D
 * @param K Number of columns in A and B
 * @param alpha Scaling factor for A @ B^T
 * @param beta Scaling factor for C
 * @param stream CUDA stream (currently unused, for future async support)
 *
 * @return cudaSuccess on success, error code otherwise
 *
 * Requirements:
 * - M must be multiple of 128
 * - N must be multiple of 256
 * - K must be multiple of 64
 * - All pointers must be valid device memory
 * - Tensors must be contiguous with specified layouts
 */
cudaError_t launch_sm100_gemm_f16_tma(void *d_A, void *d_B, void *d_C,
                                      void *d_D, int M, int N, int K,
                                      float alpha, float beta,
                                      cudaStream_t stream = 0);

#ifdef __cplusplus
}
#endif
