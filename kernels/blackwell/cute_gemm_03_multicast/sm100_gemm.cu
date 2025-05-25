// sm100_gemm_kernel.cu - CUDA kernel implementation with TMA Multicast
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

// Shared storage structure with TMA barriers for multicast
template <class TypeA, class TypeB, class ASmemLayout, class BSmemLayout>
struct SharedStorage {
  alignas(128) cute::ArrayEngine<TypeA, cute::cosize_v<ASmemLayout>> A;
  alignas(128) cute::ArrayEngine<TypeB, cute::cosize_v<BSmemLayout>> B;
  alignas(16) cute::uint64_t
      mma_barrier; // Barrier to track MMA computation on SMEM
  alignas(16) cute::uint64_t
      tma_barrier; // Barrier to track TMA data transfers to SMEM
  alignas(16) cute::uint32_t tmem_base_ptr; // Base pointer for TMEM allocation

  CUTE_DEVICE constexpr auto tensor_sA() {
    return make_tensor(make_smem_ptr(A.begin()), ASmemLayout{});
  }
  CUTE_DEVICE constexpr auto tensor_sB() {
    return make_tensor(make_smem_ptr(B.begin()), BSmemLayout{});
  }
};

// Device kernel with TMA Multicast
template <class SharedStorage, class ATensor, class BTensor, class CTensor,
          class DTensor, class MmaTiler_MNK, class TiledMMA,
          class ClusterShape_MNK, class TmaAtomA, class TmaAtomB, class Alpha,
          class Beta>
__global__ static void gemm_device_tma_multicast(
    ATensor mA, BTensor mB, CTensor mC, DTensor mD, MmaTiler_MNK mma_tiler,
    TiledMMA tiled_mma, ClusterShape_MNK cluster_shape,
    CUTE_GRID_CONSTANT TmaAtomA const tma_atom_A,
    CUTE_GRID_CONSTANT TmaAtomB const tma_atom_B, Alpha alpha, Beta beta) {
  // Step 1: The Prologue
  // The CTA layout within the Cluster: (V,M,N,K) -> CTA idx
  Layout cluster_layout_vmnk = tiled_divide(
      make_layout(cluster_shape), make_tile(typename TiledMMA::AtomThrID{}));

  // Construct the MMA grid coordinate from the CTA grid coordinate
  auto mma_coord_vmnk = make_coord(
      blockIdx.x % size<0>(cluster_layout_vmnk), // Peer CTA coordinate
      blockIdx.x / size<0>(cluster_layout_vmnk), //    MMA-M coordinate
      blockIdx.y,                                //    MMA-N coordinate
      _);                                        //    MMA-K coordinate

  auto mma_coord = select<1, 2, 3>(mma_coord_vmnk);
  Tensor gA = local_tile(mA, mma_tiler, mma_coord, Step<_1, X, _1>{});
  Tensor gB = local_tile(mB, mma_tiler, mma_coord, Step<X, _1, _1>{});
  Tensor gC = local_tile(mC, mma_tiler, mma_coord, Step<_1, _1, X>{});
  Tensor gD = local_tile(mD, mma_tiler, mma_coord, Step<_1, _1, X>{});

  // SMEM allocation
  extern __shared__ char shared_memory[];
  SharedStorage &shared_storage =
      *reinterpret_cast<SharedStorage *>(shared_memory);

  Tensor tCsA = shared_storage.tensor_sA();
  Tensor tCsB = shared_storage.tensor_sB();

  // MMA partitioning
  auto mma_v = get<0>(mma_coord_vmnk);
  ThrMMA cta_mma = tiled_mma.get_slice(mma_v);
  Tensor tCgA = cta_mma.partition_A(gA);
  Tensor tCgB = cta_mma.partition_B(gB);
  Tensor tCgC = cta_mma.partition_C(gC);
  Tensor tCgD = cta_mma.partition_C(gD);

  // Fragment allocation
  Tensor tCrA = cta_mma.make_fragment_A(tCsA);
  Tensor tCrB = cta_mma.make_fragment_B(tCsB);
  Tensor tCtAcc = cta_mma.make_fragment_C(tCgC);

  uint32_t elect_one_thr = cute::elect_one_sync();
  uint32_t elect_one_warp = (threadIdx.x / 32 == 0);

  using TmemAllocator = cute::TMEM::Allocator1Sm;
  TmemAllocator tmem_allocator{};

  // TMEM allocation
  if (elect_one_warp) {
    tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns,
                            &shared_storage.tmem_base_ptr);
  }
  __syncthreads();
  tCtAcc.data() = shared_storage.tmem_base_ptr;

  // TMA Multicast Setup
  // Construct the CTA-in-Cluster coordinate for multicasting
  auto cta_in_cluster_coord_vmnk =
      cluster_layout_vmnk.get_flat_coord(int(cute::block_rank_in_cluster()));

  // TMA partitioning with multicast support
  // Each CTA with the same m-coord will load a portion of A (multicast along N)
  // Each CTA with the same n-coord will load a portion of B (multicast along M)
  // Multicast behavior for CTA coordination:
  //   A multicast: same M-coord CTAs share A data
  //   B multicast: same N-coord CTAs share B data

  // Project the cluster_layout for tma_A along the N-modes (multicast along N)
  auto [tAgA, tAsA] = tma_partition(
      tma_atom_A,
      get<2>(cta_in_cluster_coord_vmnk), // The CTA coordinate along N mode of
                                         // the cluster
      make_layout(size<2>(
          cluster_layout_vmnk)), // The CTA layout along N mode of the cluster
      group_modes<0, 3>(tCsA), group_modes<0, 3>(tCgA));

  // Project the cluster_layout for tma_B along the M-modes (multicast along M)
  auto [tBgB, tBsB] = tma_partition(
      tma_atom_B,
      get<1>(cta_in_cluster_coord_vmnk), // The CTA coordinate along M mode of
                                         // the cluster
      make_layout(size<1>(
          cluster_layout_vmnk)), // The CTA layout along M mode of the cluster
      group_modes<0, 3>(tCsB), group_modes<0, 3>(tCgB));

  // Create multicast masks for coordinated TMA operations
  // Project the cluster_layout and cta_coord along the N-mode to determine the
  // multicast mask for A
  uint16_t tma_mcast_mask_a = create_tma_multicast_mask<2>(
      cluster_layout_vmnk, cta_in_cluster_coord_vmnk);
  // Project the cluster_layout and cta_coord along the M-mode to determine the
  // multicast mask for B
  uint16_t tma_mcast_mask_b = create_tma_multicast_mask<1>(
      cluster_layout_vmnk, cta_in_cluster_coord_vmnk);
  // Project the cluster_layout and cta_coord along the VM + VN-modes to
  // determine the multicast mask for C
  uint16_t mma_mcast_mask_c =
      create_tma_multicast_mask<0, 1>(cluster_layout_vmnk,
                                      cta_in_cluster_coord_vmnk) |
      create_tma_multicast_mask<0, 2>(cluster_layout_vmnk,
                                      cta_in_cluster_coord_vmnk);

  // Calculate total bytes that TMA will transfer each tile to track completion
  int tma_transaction_bytes =
      sizeof(make_tensor_like(tAsA)) + sizeof(make_tensor_like(tBsB));

  // Barrier initialization for multicast coordination
  if (elect_one_warp && elect_one_thr) {
    // The number of CTAs that participate in multicast operation with this CTA
    // (for both A and B matrices)
    int num_mcast_participants =
        size<1>(cluster_layout_vmnk) + size<2>(cluster_layout_vmnk) - 1;
    cute::initialize_barrier(shared_storage.mma_barrier,
                             num_mcast_participants);
    cute::initialize_barrier(shared_storage.tma_barrier, 1);
  }
  int mma_barrier_phase_bit = 0;
  int tma_barrier_phase_bit = 0;
  cute::cluster_sync(); // Make sure all threads across all CTAs in Cluster
                        // observe barrier initialization

  // Step 2: The Mainloop with TMA Multicast
  tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

  for (int k_tile = 0; k_tile < size<3>(tCgA); ++k_tile) {
    // Step 2a: TMA Multicast Load Operations
    // Execute asynchronous TMA loads with multicast masks
    if (elect_one_warp && elect_one_thr) {
      cute::set_barrier_transaction_bytes(shared_storage.tma_barrier,
                                          tma_transaction_bytes);

      // TMA Load with multicast for A matrix (multicast along N dimension)
      copy(tma_atom_A.with(shared_storage.tma_barrier, tma_mcast_mask_a),
           tAgA(_, k_tile), tAsA);

      // TMA Load with multicast for B matrix (multicast along M dimension)
      copy(tma_atom_B.with(shared_storage.tma_barrier, tma_mcast_mask_b),
           tBgB(_, k_tile), tBsB);
    }

    // Step 2b: Wait for TMA multicast loads and execute MMAs
    // Wait for TMA loads to SMEM to complete across all participating CTAs
    cute::wait_barrier(shared_storage.tma_barrier, tma_barrier_phase_bit);
    tma_barrier_phase_bit ^= 1;

    // Execute MMAs with multicast coordination
    if (elect_one_warp) {
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCtAcc);
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
      }
      // Ensure MMAs are completed across all multicasting CTAs
      cutlass::arch::umma_arrive_multicast(&shared_storage.mma_barrier,
                                           mma_mcast_mask_c);
    }

    // Wait MMAs to complete to avoid overwriting the A and B SMEM
    cute::wait_barrier(shared_storage.mma_barrier, mma_barrier_phase_bit);
    mma_barrier_phase_bit ^= 1;
  }

  // Step 3: The Epilogue
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
  copy(tiled_t2r_copy, tDtAcc, tDrAcc);

  // AXPBY and store result
  axpby(alpha, tDrAcc, beta, tDrC);
  copy(tDrC, tDgD);

  __syncthreads();

  // Cleanup TMEM
  if (elect_one_warp) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(shared_storage.tmem_base_ptr,
                        TmemAllocator::Sm100TmemCapacityColumns);
  }
}

// Host function that creates TMA multicast descriptors and launches the kernel
cudaError_t launch_sm100_gemm_f16_tma_multicast(
    void *d_A, void *d_B, void *d_C, void *d_D, int M, int N, int K,
    float alpha, float beta, cudaStream_t stream, int cluster_m,
    int cluster_n) // Configurable cluster shape
{
  // Define types
  using TypeA = cutlass::half_t;
  using TypeB = cutlass::half_t;
  using TypeC = float;
  using TypeD = float;

  // Create layouts (K-major for A and B, N-major for C and D)
  auto layout_A = make_layout(make_shape(M, K), make_stride(K, Int<1>{}));
  auto layout_B = make_layout(make_shape(N, K), make_stride(K, Int<1>{}));
  auto layout_C = make_layout(make_shape(M, N), make_stride(N, Int<1>{}));
  auto layout_D = layout_C;

  // Create CuTe tensors
  auto mA =
      make_tensor(make_gmem_ptr(reinterpret_cast<TypeA *>(d_A)), layout_A);
  auto mB =
      make_tensor(make_gmem_ptr(reinterpret_cast<TypeB *>(d_B)), layout_B);
  auto mC =
      make_tensor(make_gmem_ptr(reinterpret_cast<TypeC *>(d_C)), layout_C);
  auto mD =
      make_tensor(make_gmem_ptr(reinterpret_cast<TypeD *>(d_D)), layout_D);

  // Create TiledMMA
  TiledMMA tiled_mma =
      make_tiled_mma(SM100_MMA_F16BF16_SS<TypeA, TypeB, TypeC, 128, 256,
                                          UMMA::Major::K, UMMA::Major::K>{});

  // Define MMA tiler sizes
  auto bM = tile_size<0>(tiled_mma);            // 128
  auto bN = tile_size<1>(tiled_mma);            // 256
  auto bK = tile_size<2>(tiled_mma) * Int<4>{}; // 64
  auto mma_tiler = make_shape(bM, bN, bK);

  // Check alignment
  if (M % int(bM) != 0 || N % int(bN) != 0 || K % int(bK) != 0) {
    return cudaErrorInvalidValue;
  }

  // Create SMEM layouts
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

  // Cluster configuration for multicast
  auto cluster_shape = make_shape(Int<1>{}, cluster_m, cluster_n);
  Layout cluster_layout_vmnk =
      tiled_divide(make_layout(cluster_shape),
                   make_tile(typename decltype(tiled_mma)::AtomThrID{}));

  // Create TMA multicast descriptors for A and B matrices
  Copy_Atom tma_atom_A = make_tma_atom(
      SM90_TMA_LOAD_MULTICAST{},   // TMA load operation with multicast
      mA,                          // Source GMEM tensor
      sA_layout,                   // Destination SMEM layout
      select<0, 2>(mma_tiler),     // MK Tiler for TMA operation
      size<2>(cluster_layout_vmnk) // The number of CTAs in the N-mode for
                                   // multicasting
  );
  Tensor mA_tma = tma_atom_A.get_tma_tensor(shape(mA));

  Copy_Atom tma_atom_B = make_tma_atom(
      SM90_TMA_LOAD_MULTICAST{},   // TMA load operation with multicast
      mB,                          // Source GMEM tensor
      sB_layout,                   // Destination SMEM layout
      select<1, 2>(mma_tiler),     // NK Tiler for TMA operation
      size<1>(cluster_layout_vmnk) // The number of CTAs in the M-mode for
                                   // multicasting
  );
  Tensor mB_tma = tma_atom_B.get_tma_tensor(shape(mB));

  // Launch parameters with cluster support
  dim3 dimBlock(128);
  dim3 dimCluster(size<0>(cluster_shape), size<1>(cluster_shape),
                  size<2>(cluster_shape));
  dim3 dimGrid(round_up(ceil_div(M, int(bM)), dimCluster.x),
               round_up(ceil_div(N, int(bN)), dimCluster.y));
  int smemBytes = sizeof(SMEMStorage);

  // Get kernel pointer
  auto *kernel_ptr =
      &gemm_device_tma_multicast<SMEMStorage, decltype(mA_tma),
                                 decltype(mB_tma), decltype(mC), decltype(mD),
                                 decltype(mma_tiler), decltype(tiled_mma),
                                 decltype(cluster_shape), decltype(tma_atom_A),
                                 decltype(tma_atom_B), float, float>;

  // Set kernel attributes
  cudaError_t error = cudaFuncSetAttribute(
      kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes);
  if (error != cudaSuccess) {
    return error;
  }

  // Launch kernel with cluster support
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster,
                                         smemBytes};
  cutlass::Status status = cutlass::launch_kernel_on_cluster(
      params, (void const *)kernel_ptr, mA_tma, mB_tma, mC, mD, mma_tiler,
      tiled_mma, cluster_shape, tma_atom_A, tma_atom_B, alpha, beta);

  return (status == cutlass::Status::kSuccess) ? cudaSuccess
                                               : cudaErrorLaunchFailure;
}

// Wrapper function that chooses between different TMA modes
cudaError_t launch_sm100_gemm_f16(void *d_A, void *d_B, void *d_C, void *d_D,
                                  int M, int N, int K, float alpha, float beta,
                                  cudaStream_t stream) {
  // Determine optimal cluster configuration based on problem size
  int cluster_m = 2; // Default cluster shape
  int cluster_n = 2;

  // For larger problems, use larger clusters to maximize multicast benefits
  if (M >= 2048 && N >= 2048) {
    cluster_m = 4;
    cluster_n = 4;
  } else if (M >= 1024 && N >= 1024) {
    cluster_m = 2;
    cluster_n = 4;
  }

  // Always use TMA multicast for best performance
  return launch_sm100_gemm_f16_tma_multicast(
      d_A, d_B, d_C, d_D, M, N, K, alpha, beta, stream, cluster_m, cluster_n);
}

#else

cudaError_t launch_sm100_gemm_f16(void *d_A, void *d_B, void *d_C, void *d_D,
                                  int M, int N, int K, float alpha, float beta,
                                  cudaStream_t stream) {
  return cudaErrorNotSupported;
}

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED
