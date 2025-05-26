// sm100_gemm.h - header for CUDA kernel with TMA Multicast
#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <utility>

#ifdef __cplusplus
extern "C" {
#endif

// Constants for alignment and validation
#define SM100_MMA_TILE_M 128
#define SM100_MMA_TILE_N 256
#define SM100_MMA_TILE_K 64
#define SM100_MAX_CLUSTER_SIZE 16

// Error code extensions for better diagnostics
typedef enum {
  SM100_SUCCESS = 0,
  SM100_ERROR_INVALID_PARAMETERS = 1,
  SM100_ERROR_UNSUPPORTED_DEVICE = 2,
  SM100_ERROR_MEMORY_ALLOCATION_FAILED = 3,
  SM100_ERROR_CLUSTER_TOO_LARGE = 4,
  SM100_ERROR_ALIGNMENT_MISMATCH = 5
} sm100_status_t;

/**
 * @brief Get optimal cluster configuration for given problem size
 *
 * Analyzes the problem characteristics and returns the optimal cluster
 * configuration to maximize TMA multicast benefits while staying within
 * hardware limits.
 *
 * @param M Number of rows in A and output matrices
 * @param N Number of rows in B and columns in output matrices
 * @param K Number of columns in A and B matrices
 * @return std::pair<int, int> Optimal (cluster_m, cluster_n) configuration
 */
#ifdef __cplusplus
std::pair<int, int> optimize_cluster_configuration(int M, int N, int K);
#endif

/**
 * @brief Validate GEMM parameters for SM100 compatibility
 *
 * Checks if the provided parameters meet SM100 requirements including
 * alignment, size limits, and overflow protection.
 *
 * @param M Number of rows in A matrix
 * @param N Number of rows in B matrix
 * @param K Number of columns in A and B matrices
 * @return bool True if parameters are valid, false otherwise
 */
#ifdef __cplusplus
bool validate_gemm_parameters(int M, int N, int K);
#endif

/**
 * @brief Launch SM100 GEMM kernel with automatic TMA Multicast optimization
 *
 * High-level interface that automatically selects optimal TMA multicast
 * configuration based on problem size. Provides the best performance for
 * most use cases.
 *
 * Key Features:
 * - Automatic cluster size optimization
 * - Enhanced error handling and validation
 * - Memory bandwidth optimization through TMA multicast
 * - Support for both small and large matrix operations
 *
 * Matrix Operation: D = alpha * A @ B^T + beta * C
 *
 * @param d_A Pointer to matrix A in device memory (M x K, FP16, K-major layout)
 * @param d_B Pointer to matrix B in device memory (N x K, FP16, K-major layout)
 * @param d_C Pointer to matrix C in device memory (M x N, FP32, N-major layout)
 * @param d_D Pointer to matrix D in device memory (M x N, FP32, N-major layout)
 * @param M Number of rows in A and output matrices (must be multiple of 128)
 * @param N Number of rows in B and columns in output (must be multiple of 256)
 * @param K Number of columns in A and B matrices (must be multiple of 64)
 * @param alpha Scaling factor for A @ B^T product
 * @param beta Scaling factor for C matrix
 * @param stream CUDA stream for asynchronous execution (0 for default stream)
 *
 * @return cudaSuccess on success, appropriate CUDA error code on failure
 *
 * Requirements:
 * - GPU must support SM100 architecture (compute capability 10.0a)
 * - All pointers must point to valid, properly aligned device memory
 * - Matrices must be contiguous with specified memory layouts
 * - CUTLASS must be compiled with SM100 support enabled
 *
 * Performance Notes:
 * - Automatically optimizes cluster size: 1x1 for small, 2x2 for medium, 4x4
 * for large matrices
 * - TMA multicast provides 2-4x speedup over non-multicast implementations for
 * large matrices
 * - Best performance achieved when dimensions are large multiples of tile sizes
 * - Memory bandwidth utilization can exceed 80% of theoretical peak
 */
cudaError_t launch_sm100_gemm_f16(void *d_A, void *d_B, void *d_C, void *d_D,
                                  int M, int N, int K, float alpha, float beta,
                                  cudaStream_t stream = 0);

/**
 * @brief Launch SM100 GEMM kernel with explicit TMA Multicast configuration
 *
 * Advanced interface that allows manual control over cluster configuration.
 * Use this when you need fine-grained control over the multicast behavior
 * or have specific performance requirements.
 *
 * TMA Multicast Behavior:
 * - A matrix data is multicast along the N dimension of the cluster
 * - B matrix data is multicast along the M dimension of the cluster
 * - Larger clusters provide better bandwidth utilization but require more
 * memory
 * - Cluster size affects load balancing and synchronization overhead
 *
 * @param d_A Pointer to matrix A in device memory
 * @param d_B Pointer to matrix B in device memory
 * @param d_C Pointer to matrix C in device memory
 * @param d_D Pointer to matrix D in device memory
 * @param M Number of rows in A and output matrices
 * @param N Number of rows in B and columns in output matrices
 * @param K Number of columns in A and B matrices
 * @param alpha Scaling factor for A @ B^T product
 * @param beta Scaling factor for C matrix
 * @param stream CUDA stream for asynchronous execution
 * @param cluster_m Cluster size in M dimension (1, 2, 4, or 8)
 * @param cluster_n Cluster size in N dimension (1, 2, 4, or 8)
 *
 * @return cudaSuccess on success, appropriate CUDA error code on failure
 *
 * Cluster Configuration Guidelines:
 * - Use 1x1 for matrices smaller than 1024x1024
 * - Use 2x2 for medium matrices (1024x1024 to 4096x4096)
 * - Use 4x4 for large matrices (4096x4096 and above)
 * - Use 2x4 or 4x2 for rectangular matrices with high aspect ratios
 * - cluster_m * cluster_n must not exceed hardware limits (typically 16)
 *
 * Advanced Usage Examples:
 * - Memory-bound workloads: Prefer smaller clusters (2x2) to reduce contention
 * - Compute-bound workloads: Use larger clusters (4x4) for maximum throughput
 * - Bandwidth-critical: Use asymmetric clusters (2x4) based on matrix shape
 */
cudaError_t launch_sm100_gemm_f16_tma_multicast(void *d_A, void *d_B, void *d_C,
                                                void *d_D, int M, int N, int K,
                                                float alpha, float beta,
                                                cudaStream_t stream = 0,
                                                int cluster_m = 2,
                                                int cluster_n = 2);

/**
 * @brief Check if SM100 GEMM operations are supported on current device
 *
 * Performs comprehensive checks for SM100 support including:
 * - Compute capability validation (requires 10.0a)
 * - CUTLASS compilation flags
 * - TMA multicast hardware support
 * - Memory subsystem compatibility
 *
 * @return bool True if SM100 GEMM is fully supported, false otherwise
 */
#ifdef __cplusplus
bool is_sm100_gemm_supported();
#endif

/**
 * @brief Get detailed SM100 device capabilities and limits
 *
 * Returns comprehensive information about SM100 capabilities including:
 * - Maximum cluster sizes supported
 * - TMEM capacity and allocation granularity
 * - TMA bandwidth characteristics
 * - Optimal tile sizes for current device
 *
 * @param[out] max_cluster_m Maximum cluster size in M dimension
 * @param[out] max_cluster_n Maximum cluster size in N dimension
 * @param[out] tmem_capacity Total TMEM capacity in bytes
 * @param[out] tma_bandwidth Peak TMA bandwidth in GB/s
 *
 * @return cudaSuccess if information retrieved successfully
 */
cudaError_t get_sm100_device_info(int *max_cluster_m, int *max_cluster_n,
                                  size_t *tmem_capacity, float *tma_bandwidth);

/**
 * @brief Estimate performance characteristics for given problem size
 *
 * Provides performance estimates and optimization recommendations without
 * actually executing the kernel. Useful for auto-tuning and performance
 * modeling.
 *
 * @param M Matrix dimension M
 * @param N Matrix dimension N
 * @param K Matrix dimension K
 * @param cluster_m Cluster size in M dimension
 * @param cluster_n Cluster size in N dimension
 * @param[out] estimated_gflops Estimated performance in GFLOPS
 * @param[out] estimated_bandwidth Estimated memory bandwidth in GB/s
 * @param[out] estimated_time Estimated execution time in milliseconds
 *
 * @return cudaSuccess if estimation completed successfully
 */
cudaError_t estimate_sm100_performance(int M, int N, int K, int cluster_m,
                                       int cluster_n, float *estimated_gflops,
                                       float *estimated_bandwidth,
                                       float *estimated_time);

/**
 * @brief Create aligned tensors optimized for SM100 TMA operations
 *
 * Allocates device memory with proper alignment and padding to maximize
 * TMA multicast efficiency. Returns pointers to optimally aligned memory
 * regions.
 *
 * @param M_orig Original M dimension (will be padded if needed)
 * @param N_orig Original N dimension (will be padded if needed)
 * @param K_orig Original K dimension (will be padded if needed)
 * @param[out] d_A_aligned Pointer to aligned A matrix memory
 * @param[out] d_B_aligned Pointer to aligned B matrix memory
 * @param[out] d_C_aligned Pointer to aligned C matrix memory
 * @param[out] d_D_aligned Pointer to aligned D matrix memory
 * @param[out] M_aligned Actual aligned M dimension
 * @param[out] N_aligned Actual aligned N dimension
 * @param[out] K_aligned Actual aligned K dimension
 *
 * @return cudaSuccess if allocation successful
 *
 * Note: Caller is responsible for freeing allocated memory using cudaFree()
 */
cudaError_t create_aligned_sm100_tensors(int M_orig, int N_orig, int K_orig,
                                         void **d_A_aligned, void **d_B_aligned,
                                         void **d_C_aligned, void **d_D_aligned,
                                         int *M_aligned, int *N_aligned,
                                         int *K_aligned);

/**
 * @brief Benchmark SM100 GEMM performance across different configurations
 *
 * Runs comprehensive performance benchmarks to find optimal cluster
 * configuration for specific problem sizes. Useful for auto-tuning
 * applications.
 *
 * @param M Matrix dimension M
 * @param N Matrix dimension N
 * @param K Matrix dimension K
 * @param num_iterations Number of timing iterations
 * @param[out] best_cluster_m Optimal cluster size in M dimension
 * @param[out] best_cluster_n Optimal cluster size in N dimension
 * @param[out] best_time Best execution time achieved
 * @param[out] best_gflops Best performance in GFLOPS
 *
 * @return cudaSuccess if benchmark completed successfully
 */
cudaError_t
benchmark_sm100_configurations(int M, int N, int K, int num_iterations,
                               int *best_cluster_m, int *best_cluster_n,
                               float *best_time, float *best_gflops);

/**
 * @brief Advanced memory layout optimization for TMA multicast
 *
 * Analyzes matrix access patterns and suggests optimal memory layouts
 * to maximize TMA multicast efficiency. Can recommend padding strategies
 * and memory alignment for best performance.
 *
 * @param M Matrix dimension M
 * @param N Matrix dimension N
 * @param K Matrix dimension K
 * @param access_pattern Expected access pattern (0=sequential, 1=strided,
 * 2=random)
 * @param[out] recommended_padding_M Recommended padding for M dimension
 * @param[out] recommended_padding_N Recommended padding for N dimension
 * @param[out] recommended_padding_K Recommended padding for K dimension
 * @param[out] memory_efficiency Expected memory efficiency (0.0 to 1.0)
 *
 * @return cudaSuccess if analysis completed successfully
 */
cudaError_t optimize_sm100_memory_layout(int M, int N, int K,
                                         int access_pattern,
                                         int *recommended_padding_M,
                                         int *recommended_padding_N,
                                         int *recommended_padding_K,
                                         float *memory_efficiency);

// Utility macros for common operations
#define SM100_ALIGN_M(m)                                                       \
  (((m) + SM100_MMA_TILE_M - 1) / SM100_MMA_TILE_M * SM100_MMA_TILE_M)
#define SM100_ALIGN_N(n)                                                       \
  (((n) + SM100_MMA_TILE_N - 1) / SM100_MMA_TILE_N * SM100_MMA_TILE_N)
#define SM100_ALIGN_K(k)                                                       \
  (((k) + SM100_MMA_TILE_K - 1) / SM100_MMA_TILE_K * SM100_MMA_TILE_K)

#define SM100_IS_ALIGNED_M(m) ((m) % SM100_MMA_TILE_M == 0)
#define SM100_IS_ALIGNED_N(n) ((n) % SM100_MMA_TILE_N == 0)
#define SM100_IS_ALIGNED_K(k) ((k) % SM100_MMA_TILE_K == 0)

// Error checking macro
#define SM100_CHECK_ERROR(call)                                                \
  do {                                                                         \
    cudaError_t error = (call);                                                \
    if (error != cudaSuccess) {                                                \
      fprintf(stderr, "SM100 Error at %s:%d - %s\n", __FILE__, __LINE__,       \
              cudaGetErrorString(error));                                      \
      return error;                                                            \
    }                                                                          \
  } while (0)

#ifdef __cplusplus
}
#endif
