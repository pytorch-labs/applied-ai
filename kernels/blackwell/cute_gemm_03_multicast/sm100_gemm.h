// sm100_gemm_kernel.h - Header file for CUDA kernel with TMA Multicast
#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Launch SM100 GEMM kernel with TMA Multicast: D = alpha * A @ B^T + beta * C
 *
 * This function automatically selects optimal TMA multicast configuration based
 * on problem size. TMA Multicast enables efficient data sharing across multiple
 * CTAs in a cluster:
 * - A matrix data is multicast along the N dimension (same M-coord CTAs share A
 * data)
 * - B matrix data is multicast along the M dimension (same N-coord CTAs share B
 * data)
 * - Reduces memory bandwidth requirements and improves cache efficiency
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
 * - GPU must support SM100 architecture and TMA multicast
 *
 * Performance Notes:
 * - TMA multicast provides significant speedup for large matrices (>1GB)
 * - Cluster size is automatically optimized: 2x2 for medium, 4x4 for large
 * matrices
 * - Best performance achieved when M, N dimensions are large multiples of tile
 * sizes
 */
cudaError_t launch_sm100_gemm_f16(void *d_A, void *d_B, void *d_C, void *d_D,
                                  int M, int N, int K, float alpha, float beta,
                                  cudaStream_t stream = 0);

/**
 * Launch SM100 GEMM kernel with explicit TMA Multicast configuration
 *
 * @param d_A Pointer to matrix A in device memory
 * @param d_B Pointer to matrix B in device memory
 * @param d_C Pointer to matrix C in device memory
 * @param d_D Pointer to matrix D in device memory
 * @param M Number of rows in A and C/D
 * @param N Number of rows in B and columns in C/D
 * @param K Number of columns in A and B
 * @param alpha Scaling factor for A @ B^T
 * @param beta Scaling factor for C
 * @param stream CUDA stream
 * @param cluster_m Cluster size in M dimension (1, 2, 4, or 8)
 * @param cluster_n Cluster size in N dimension (1, 2, 4, or 8)
 *
 * @return cudaSuccess on success, error code otherwise
 *
 * Advanced Usage:
 * - Use smaller clusters (2x2) for memory-bound workloads
 * - Use larger clusters (4x4 or 8x8) for compute-bound workloads with large
 * matrices
 * - cluster_m * cluster_n should not exceed maximum cluster size for the GPU
 */
cudaError_t launch_sm100_gemm_f16_tma_multicast(void *d_A, void *d_B, void *d_C,
                                                void *d_D, int M, int N, int K,
                                                float alpha, float beta,
                                                cudaStream_t stream = 0,
                                                int cluster_m = 2,
                                                int cluster_n = 2);

#ifdef __cplusplus
}
#endif
