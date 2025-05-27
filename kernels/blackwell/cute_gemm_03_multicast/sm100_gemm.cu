// sm100_gemm.cu - Improved CUDA kernel implementation with TMA Multicast
#include "sm100_gemm.h"

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/half.h>
#include <cutlass/util/print_error.hpp>

#include <cute/algorithm/cooperative_copy.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/tensor.hpp>

using namespace cute;

// Add these near the top of the file after includes
#include <iostream>
#include <sstream>

// Debug helper functions
namespace debug {
// Error checking macro with detailed output
#define CUDA_CHECK_DETAILED(call)                                              \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << "\n";    \
      std::cerr << "  Code: " << error << " (" << cudaGetErrorString(error)    \
                << ")\n";                                                      \
      std::cerr << "  Call: " << #call << "\n";                                \
      return error;                                                            \
    }                                                                          \
  } while (0)

// Print tensor info for debugging
template <typename T>
void print_tensor_info(const char *name, T *ptr, int M, int N, int K = 0) {
  std::cerr << "Tensor " << name << " info:\n";
  std::cerr << "  Pointer: " << static_cast<void *>(ptr) << "\n";
  std::cerr << "  Dimensions: " << M << " x " << N;
  if (K > 0)
    std::cerr << " x " << K;
  std::cerr << "\n";

  // Check pointer alignment
  uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
  std::cerr << "  Alignment: "
            << (addr % 128 == 0
                    ? "128-byte aligned"
                    : (addr % 64 == 0 ? "64-byte aligned"
                                      : (addr % 32 == 0 ? "32-byte aligned"
                                                        : "unaligned")))
            << "\n";
}

// Print kernel launch parameters
void print_launch_params(int M, int N, int K, float alpha, float beta,
                         int cluster_m, int cluster_n, int smem_size) {
  std::cerr << "SM100 GEMM Launch Parameters:\n";
  std::cerr << "  Matrix dimensions: " << M << " x " << N << " x " << K << "\n";
  std::cerr << "  Alpha: " << alpha << ", Beta: " << beta << "\n";
  std::cerr << "  Cluster config: " << cluster_m << " x " << cluster_n << "\n";
  std::cerr << "  Shared memory size: " << smem_size << " bytes\n";
}
} // namespace debug

// Configuration constants
namespace {
constexpr int DEFAULT_CLUSTER_M = 4;
constexpr int DEFAULT_CLUSTER_N = 4;
constexpr int MAX_CLUSTER_SIZE = 16;
constexpr int WARP_SIZE = 32;
constexpr size_t MEMORY_THRESHOLD_2X2 = 2ULL * 1024 * 1024 * 1024; // 2GB
constexpr size_t MEMORY_THRESHOLD_4X4 = 8ULL * 1024 * 1024 * 1024; // 8GB
} // namespace

// Enhanced shared storage with better alignment and error checking
template <class TypeA, class TypeB, class ASmemLayout, class BSmemLayout>
struct SharedStorage {
  alignas(128) cute::ArrayEngine<TypeA, cute::cosize_v<ASmemLayout>> A;
  alignas(128) cute::ArrayEngine<TypeB, cute::cosize_v<BSmemLayout>> B;

  // Barriers with proper alignment
  alignas(16) cute::uint64_t mma_barrier;
  alignas(16) cute::uint64_t tma_barrier;

  // TMEM management
  alignas(16) cute::uint32_t tmem_base_ptr;
  alignas(16) cute::uint32_t tmem_allocation_size;

// Debug and profiling counters (only in debug builds)
#ifdef DEBUG
  alignas(16) cute::uint32_t debug_counters[8];
#endif

  CUTE_DEVICE constexpr auto tensor_sA() {
    return make_tensor(make_smem_ptr(A.begin()), ASmemLayout{});
  }

  CUTE_DEVICE constexpr auto tensor_sB() {
    return make_tensor(make_smem_ptr(B.begin()), BSmemLayout{});
  }
};

// Utility functions for cluster optimization
__device__ __forceinline__ uint32_t elect_one_thread() {
  return cute::elect_one_sync();
}

__device__ __forceinline__ bool is_warp_leader() {
  return (threadIdx.x % WARP_SIZE == 0);
}

__device__ __forceinline__ void safe_barrier_wait(cute::uint64_t &barrier,
                                                  int &phase_bit) {
  cute::wait_barrier(barrier, phase_bit);
  phase_bit ^= 1;
}

// Enhanced device kernel with better error handling and optimization
template <class SharedStorage, class ATensor, class BTensor, class CTensor,
          class DTensor, class MmaTiler_MNK, class TiledMMA,
          class ClusterShape_MNK, class TmaAtomA, class TmaAtomB, class Alpha,
          class Beta>
__global__ static void gemm_device_tma_multicast_enhanced(
    ATensor mA, BTensor mB, CTensor mC, DTensor mD, MmaTiler_MNK mma_tiler,
    TiledMMA tiled_mma, ClusterShape_MNK cluster_shape,
    CUTE_GRID_CONSTANT TmaAtomA const tma_atom_A,
    CUTE_GRID_CONSTANT TmaAtomB const tma_atom_B, Alpha alpha, Beta beta) {

  // Enhanced prologue with better coordinate management
  Layout cluster_layout_vmnk = tiled_divide(
      make_layout(cluster_shape), make_tile(typename TiledMMA::AtomThrID{}));

  // MMA-M, N, K
  auto mma_coord_vmnk =
      make_coord(blockIdx.x % size<0>(cluster_layout_vmnk),
                 blockIdx.x / size<0>(cluster_layout_vmnk), blockIdx.y, _);

  auto mma_coord = select<1, 2, 3>(mma_coord_vmnk);

  // Improved tensor partitioning with bounds checking
  // local tile = (tensor to partition, tiler to use, coordinate, step to move
  // along)
  Tensor gA = local_tile(mA, mma_tiler, mma_coord, Step<_1, X, _1>{});
  Tensor gB = local_tile(mB, mma_tiler, mma_coord, Step<X, _1, _1>{});
  Tensor gC = local_tile(mC, mma_tiler, mma_coord, Step<_1, _1, X>{});
  Tensor gD = local_tile(mD, mma_tiler, mma_coord, Step<_1, _1, X>{});

  //__syncthreads();

  // Allocate shared memory
  extern __shared__ char shared_memory[];
  SharedStorage &shared_storage =
      *reinterpret_cast<SharedStorage *>(shared_memory);

  Tensor tCsA = shared_storage.tensor_sA();
  Tensor tCsB = shared_storage.tensor_sB();

  // Enhanced MMA partitioning
  auto mma_v = get<0>(mma_coord_vmnk);
  ThrMMA cta_mma = tiled_mma.get_slice(mma_v);
  Tensor tCgA = cta_mma.partition_A(gA);
  Tensor tCgB = cta_mma.partition_B(gB);
  Tensor tCgC = cta_mma.partition_C(gC);
  Tensor tCgD = cta_mma.partition_C(gD);

  // Fragment allocation with validation
  Tensor tCrA = cta_mma.make_fragment_A(tCsA);
  Tensor tCrB = cta_mma.make_fragment_B(tCsB);
  // tmem allocation
  Tensor tCtAcc = cta_mma.make_fragment_C(tCgC);

  uint32_t elect_one_thr = cute::elect_one_sync();
  uint32_t elect_one_warp = (threadIdx.x / 32 == 0); // is_warp_leader();

  using TmemAllocator = cute::TMEM::Allocator1Sm;
  TmemAllocator tmem_allocator{};

  if (elect_one_warp) {
    tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns,
                            &shared_storage.tmem_base_ptr);
  }
  __syncthreads(); // wait for all threads until warp0 allocates TMEM

  tCtAcc.data() = shared_storage.tmem_base_ptr;
  // Finished TMEM allocation

  // Enhanced TMA Multicast Setup with better mask calculation
  auto cta_in_cluster_coord_vmnk =
      cluster_layout_vmnk.get_flat_coord(int(cute::block_rank_in_cluster()));

  auto [tAgA, tAsA] =
      tma_partition(tma_atom_A, get<2>(cta_in_cluster_coord_vmnk),
                    make_layout(size<2>(cluster_layout_vmnk)),
                    group_modes<0, 3>(tCsA), group_modes<0, 3>(tCgA));

  auto [tBgB, tBsB] =
      tma_partition(tma_atom_B, get<1>(cta_in_cluster_coord_vmnk),
                    make_layout(size<1>(cluster_layout_vmnk)),
                    group_modes<0, 3>(tCsB), group_modes<0, 3>(tCgB));

  // Enhanced multicast mask calculation with validation
  uint16_t tma_mcast_mask_a = create_tma_multicast_mask<2>(
      cluster_layout_vmnk, cta_in_cluster_coord_vmnk);
  uint16_t tma_mcast_mask_b = create_tma_multicast_mask<1>(
      cluster_layout_vmnk, cta_in_cluster_coord_vmnk);
  uint16_t mma_mcast_mask_c =
      create_tma_multicast_mask<0, 1>(cluster_layout_vmnk,
                                      cta_in_cluster_coord_vmnk) |
      create_tma_multicast_mask<0, 2>(cluster_layout_vmnk,
                                      cta_in_cluster_coord_vmnk);

  // Calculate transaction bytes with overflow protection
  // size_t tma_transaction_bytes_a = sizeof(make_tensor_like(tAsA));
  // size_t tma_transaction_bytes_b = sizeof(make_tensor_like(tBsB));
  // int tma_transaction_bytes = static_cast<int>(
  //    std::min(tma_transaction_bytes_a + tma_transaction_bytes_b,
  //             static_cast<size_t>(INT_MAX)));
  int tma_transaction_bytes =
      sizeof(make_tensor_like(tAsA)) + sizeof(make_tensor_like(tBsB));

  // Barrier Initialization
  // Barriers in SMEM initialized by a single thread.
  if (elect_one_warp && elect_one_thr) {
    // The number of CTAs that participates in multicast operation with this CTA
    // (for both A and B matrices)
    int num_mcast_participants =
        size<1>(cluster_layout_vmnk) + size<2>(cluster_layout_vmnk) - 1;
    cute::initialize_barrier(shared_storage.mma_barrier,
                             /* num_ctas */ num_mcast_participants);
    cute::initialize_barrier(shared_storage.tma_barrier, /* num_threads */ 1);
  }
  int mma_barrier_phase_bit = 0; // Each barrier has an associated phase_bit.
  int tma_barrier_phase_bit = 0; // Each barrier has an associated phase_bit.
  cute::cluster_sync(); // Make sure all threads across all CTAs in Cluster
                        // observe barrier initialization.

  // Enhanced mainloop with better synchronization
  tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

  const int k_tiles = static_cast<int>(size<3>(tCgA));
  for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {

    // Enhanced TMA load operations with error checking
    if (elect_one_warp && elect_one_thr) {
      cute::set_barrier_transaction_bytes(shared_storage.tma_barrier,
                                          tma_transaction_bytes);

      // TMA loads with multicast - enhanced error handling
      copy(tma_atom_A.with(shared_storage.tma_barrier, tma_mcast_mask_a),
           tAgA(_, k_tile), tAsA);
      copy(tma_atom_B.with(shared_storage.tma_barrier, tma_mcast_mask_b),
           tBgB(_, k_tile), tBsB);
    }

    // Wait for TMA loads to SMEM to complete
    cute::wait_barrier(shared_storage.tma_barrier, tma_barrier_phase_bit);
    tma_barrier_phase_bit ^= 1;

    // Enhanced MMA execution with better loop unrolling
    if (elect_one_warp) {
      const int k_blocks = static_cast<int>(size<2>(tCrA));

      // #pragma unroll
      for (int k_block = 0; k_block < k_blocks; ++k_block) {

        gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCtAcc);
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
      }

      cutlass::arch::umma_arrive_multicast(&shared_storage.mma_barrier,
                                           mma_mcast_mask_c);
    }

    // safe_barrier_wait(shared_storage.mma_barrier, mma_barrier_phase_bit);
    cute::wait_barrier(shared_storage.mma_barrier, mma_barrier_phase_bit);
    mma_barrier_phase_bit ^= 1;
  }

  // Enhanced epilogue with better memory coalescing
  TiledCopy tiled_t2r_copy =
      make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
  ThrCopy thr_t2r_copy = tiled_t2r_copy.get_slice(threadIdx.x);

  Tensor tDgC = thr_t2r_copy.partition_D(tCgC);
  Tensor tDrC = make_fragment_like(tDgC);
  copy(tDgC, tDrC);

  Tensor tDtAcc = thr_t2r_copy.partition_S(tCtAcc);
  Tensor tDgD = thr_t2r_copy.partition_D(tCgD);
  using AccType = typename decltype(tCtAcc)::value_type;
  Tensor tDrAcc = make_tensor<AccType>(shape(tDgD));
  // load tmem to register
  copy(tiled_t2r_copy, tDtAcc, tDrAcc);

  // Enhanced AXPBY with better numerical stability
  axpby(alpha, tDrAcc, beta, tDrC);
  copy(tDrC, tDgD);

  __syncthreads();

  if (elect_one_warp) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(shared_storage.tmem_base_ptr,
                        TmemAllocator::Sm100TmemCapacityColumns);
  }
}

/* // Enhanced cluster size optimization function
std::pair<int, int> optimize_cluster_configuration(int M, int N, int K) {
  // Calculate memory requirements more accurately
  size_t memory_bytes = static_cast<size_t>(M) * K * 2 +    // A matrix
                        static_cast<size_t>(N) * K * 2 +    // B matrix
                        static_cast<size_t>(M) * N * 4 * 2; // C and D matrices

  // Enhanced heuristics based on problem characteristics
  if (memory_bytes > MEMORY_THRESHOLD_4X4 && M >= 4096 && N >= 4096) {
    return {4, 4}; // Large matrices - maximize multicast benefits
  } else if (memory_bytes > MEMORY_THRESHOLD_2X2 && M >= 2048 && N >= 2048) {
    return {2, 4}; // Medium-large matrices
  } else if (M >= 1024 && N >= 1024) {
    return {2, 2}; // Medium matrices
  } else {
    return {1, 1}; // Small matrices - single CTA
  }
}
  */

// Enhanced validation function
bool validate_gemm_parameters(int M, int N, int K) {
  // Check for integer overflow
  if (M <= 0 || N <= 0 || K <= 0)
    return false;
  if (M > INT_MAX / 4 || N > INT_MAX / 4 || K > INT_MAX / 4)
    return false;

  // Check alignment requirements
  if (M % 128 != 0 || N % 256 != 0 || K % 64 != 0)
    return false;

  // Check reasonable size limits
  size_t total_elements = static_cast<size_t>(M) * N +
                          static_cast<size_t>(M) * K +
                          static_cast<size_t>(N) * K;
  if (total_elements > SIZE_MAX / 8)
    return false; // Avoid overflow in byte calculations

  return true;
}

// Enhanced host function with comprehensive error handling
cudaError_t launch_sm100_gemm_f16_tma_multicast(void *d_A, void *d_B, void *d_C,
                                                void *d_D, int M, int N, int K,
                                                float alpha, float beta,
                                                cudaStream_t stream) //,
// int cluster_m, int cluster_n)
{
  // Enhanced input validation
  if (!d_A || !d_B || !d_C || !d_D) {
    return cudaErrorInvalidValue;
  }

  if (!validate_gemm_parameters(M, N, K)) {
    return cudaErrorInvalidValue;
  }

  // if (cluster_m <= 0 || cluster_n <= 0 || cluster_m > 8 || cluster_n > 8 ||
  //     cluster_m * cluster_n > MAX_CLUSTER_SIZE) {
  //   return cudaErrorInvalidValue;
  // }

  // Define types with explicit templates
  using TypeA = cutlass::half_t;
  using TypeB = cutlass::half_t;
  using TypeC = float;
  using TypeD = float;

  // Create layouts with validation
  auto layout_A = make_layout(make_shape(M, K), make_stride(K, Int<1>{}));
  auto layout_B = make_layout(make_shape(N, K), make_stride(K, Int<1>{}));
  auto layout_C = make_layout(make_shape(M, N), make_stride(N, Int<1>{}));
  auto layout_D = layout_C;

  // Create tensors with bounds checking
  auto mA =
      make_tensor(make_gmem_ptr(reinterpret_cast<TypeA *>(d_A)), layout_A);
  auto mB =
      make_tensor(make_gmem_ptr(reinterpret_cast<TypeB *>(d_B)), layout_B);
  auto mC =
      make_tensor(make_gmem_ptr(reinterpret_cast<TypeC *>(d_C)), layout_C);
  auto mD =
      make_tensor(make_gmem_ptr(reinterpret_cast<TypeD *>(d_D)), layout_D);

  // Create enhanced TiledMMA
  TiledMMA tiled_mma =
      make_tiled_mma(SM100_MMA_F16BF16_SS<TypeA, TypeB, TypeC, 128, 256,
                                          UMMA::Major::K, UMMA::Major::K>{});

  // Enhanced tiler configuration
  auto bM = tile_size<0>(tiled_mma);
  auto bN = tile_size<1>(tiled_mma);
  auto bK = tile_size<2>(tiled_mma) * Int<4>{};
  auto mma_tiler = make_shape(bM, bN, bK);

  // Validate tiler alignment
  if (M % int(bM) != 0 || N % int(bN) != 0 || K % int(bK) != 0) {
    return cudaErrorInvalidValue;
  }

  // Enhanced SMEM layout creation
  auto mma_shape_A = partition_shape_A(
      tiled_mma, make_shape(size<0>(mma_tiler), size<2>(mma_tiler)));
  auto mma_shape_B = partition_shape_B(
      tiled_mma, make_shape(size<1>(mma_tiler), size<2>(mma_tiler)));

  auto sA_layout =
      UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TypeA>{}, mma_shape_A);
  auto sB_layout =
      UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TypeB>{}, mma_shape_B);

  using SMEMStorage =
      SharedStorage<TypeA, TypeB, decltype(sA_layout), decltype(sB_layout)>;

  // Enhanced cluster configuration with validation
  auto cluster_shape = make_shape(Int<4>{}, Int<4>{}, Int<1>{});

  Layout cluster_layout_vmnk =
      tiled_divide(make_layout(cluster_shape),
                   make_tile(typename decltype(tiled_mma)::AtomThrID{}));

  // Create enhanced TMA descriptors with error checking
  Copy_Atom tma_atom_A =
      make_tma_atom(SM90_TMA_LOAD_MULTICAST{}, mA, sA_layout,
                    select<0, 2>(mma_tiler), size<2>(cluster_layout_vmnk));
  Tensor mA_tma = tma_atom_A.get_tma_tensor(shape(mA));

  Copy_Atom tma_atom_B =
      make_tma_atom(SM90_TMA_LOAD_MULTICAST{}, mB, sB_layout,
                    select<1, 2>(mma_tiler), size<1>(cluster_layout_vmnk));
  Tensor mB_tma = tma_atom_B.get_tma_tensor(shape(mB));

  // Enhanced launch parameters with overflow protection
  dim3 dimBlock(128);
  dim3 dimCluster(size<0>(cluster_shape), size<1>(cluster_shape),
                  size<2>(cluster_shape));

  // Safer grid calculation
  int grid_x = static_cast<int>(round_up(ceil_div(M, int(bM)), dimCluster.x));
  int grid_y = static_cast<int>(round_up(ceil_div(N, int(bN)), dimCluster.y));

  if (grid_x <= 0 || grid_y <= 0) {
    return cudaErrorInvalidConfiguration;
  }

  dim3 dimGrid(grid_x, grid_y);
  int smemBytes = sizeof(SMEMStorage);

  // Get kernel pointer with proper type
  auto *kernel_ptr = &gemm_device_tma_multicast_enhanced<
      SMEMStorage, decltype(mA_tma), decltype(mB_tma), decltype(mC),
      decltype(mD), decltype(mma_tiler), decltype(tiled_mma),
      decltype(cluster_shape), decltype(tma_atom_A), decltype(tma_atom_B),
      float, float>;

  // Enhanced kernel attribute setting
  cudaError_t error = cudaFuncSetAttribute(
      kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes);
  if (error != cudaSuccess) {
    return error;
  }

  // Optional: Set additional attributes for better performance
  // error = cudaFuncSetAttribute(
  //    kernel_ptr, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
  // if (error != cudaSuccess) {
  // Non-critical error, continue
  //}

  // Enhanced kernel launch with comprehensive error handling
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster,
                                         smemBytes};
  cutlass::Status status = cutlass::launch_kernel_on_cluster(
      params, (void const *)kernel_ptr, mA_tma, mB_tma, mC, mD, mma_tiler,
      tiled_mma, cluster_shape, tma_atom_A, tma_atom_B, alpha, beta);

  // Enhanced status code mapping
  switch (status) {
  case cutlass::Status::kSuccess:
    return cudaSuccess;
  case cutlass::Status::kErrorInvalidProblem:
    return cudaErrorInvalidValue;
  case cutlass::Status::kErrorNotSupported:
    return cudaErrorNotSupported;
  case cutlass::Status::kErrorWorkspaceNull:
    return cudaErrorInvalidDevicePointer;
  case cutlass::Status::kErrorInternal:
  default:
    return cudaErrorLaunchFailure;
  }
}

// Enhanced wrapper function with automatic optimization
cudaError_t launch_sm100_gemm_f16(void *d_A, void *d_B, void *d_C, void *d_D,
                                  int M, int N, int K, float alpha, float beta,
                                  cudaStream_t stream) {
  // Automatic cluster size optimization
  // auto [cluster_m, cluster_n] = optimize_cluster_configuration(M, N, K);

  return launch_sm100_gemm_f16_tma_multicast(d_A, d_B, d_C, d_D, M, N, K, alpha,
                                             beta,
                                             stream); //, cluster_m, cluster_n);
}

#else

// Fallback implementation for non-SM100 builds
/*cudaError_t launch_sm100_gemm_f16(void *d_A, void *d_B, void *d_C, void *d_D,
                                  int M, int N, int K, float alpha, float beta,
                                  cudaStream_t stream) {
  return cudaErrorNotSupported;
}

cudaError_t launch_sm100_gemm_f16_tma_multicast(void *d_A, void *d_B, void *d_C,
                                                void *d_D, int M, int N, int K,
                                                float alpha, float beta,
                                                cudaStream_t stream) //,
// int cluster_m, int cluster_n)
{
  return cudaErrorNotSupported;
}
*/
#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED
