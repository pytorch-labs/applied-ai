// sm100_gemm_kernel.cu - CUDA kernel implementation
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

// Shared storage structure
template <class TypeA, class TypeB, class ASmemLayout, class BSmemLayout>
struct SharedStorage {
  alignas(128) cute::ArrayEngine<TypeA, cute::cosize_v<ASmemLayout>> A;
  alignas(128) cute::ArrayEngine<TypeB, cute::cosize_v<BSmemLayout>> B;
  alignas(16) cute::uint64_t mma_barrier;
  alignas(16) cute::uint32_t tmem_base_ptr;

  CUTE_DEVICE constexpr auto tensor_sA() {
    return make_tensor(make_smem_ptr(A.begin()), ASmemLayout{});
  }
  CUTE_DEVICE constexpr auto tensor_sB() {
    return make_tensor(make_smem_ptr(B.begin()), BSmemLayout{});
  }
};

// Device kernel
template <class SharedStorage, class ATensor, class BTensor, class CTensor,
          class DTensor, class MmaTiler_MNK, class TiledMMA,
          class ClusterShape_MNK, class Alpha, class Beta>
__global__ static void
gemm_device(ATensor mA, BTensor mB, CTensor mC, DTensor mD,
            MmaTiler_MNK mma_tiler, TiledMMA tiled_mma,
            ClusterShape_MNK cluster_shape, Alpha alpha, Beta beta) {
  // Step 1: The Prologue
  Layout cluster_layout_vmnk = tiled_divide(
      make_layout(cluster_shape), make_tile(typename TiledMMA::AtomThrID{}));

  auto mma_coord_vmnk =
      make_coord(blockIdx.x % size<0>(cluster_layout_vmnk),
                 blockIdx.x / size<0>(cluster_layout_vmnk), blockIdx.y, _);

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

  // Barrier initialization
  if (elect_one_warp && elect_one_thr) {
    cute::initialize_barrier(shared_storage.mma_barrier, 1);
  }
  int mma_barrier_phase_bit = 0;
  __syncthreads();

  // Step 2: The Mainloop
  tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

  for (int k_tile = 0; k_tile < size<3>(tCgA); ++k_tile) {
    // Load A and B tiles
    cooperative_copy<128>(threadIdx.x, tCgA(_, _, _, k_tile), tCsA);
    cooperative_copy<128>(threadIdx.x, tCgB(_, _, _, k_tile), tCsB);

    __syncthreads();

    // Execute MMAs
    if (elect_one_warp) {
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCtAcc);
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
      }
      cutlass::arch::umma_arrive(&shared_storage.mma_barrier);
    }
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

// Host function that launches the kernel
cudaError_t launch_sm100_gemm_f16(void *d_A, void *d_B, void *d_C, void *d_D,
                                  int M, int N, int K, float alpha, float beta,
                                  cudaStream_t stream) {
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

  // Cluster configuration
  auto cluster_shape = make_shape(Int<1>{}, Int<1>{}, Int<1>{});

  // Launch parameters
  dim3 dimBlock(128);
  dim3 dimCluster(size<0>(cluster_shape), size<1>(cluster_shape),
                  size<2>(cluster_shape));
  dim3 dimGrid(ceil_div(M, int(bM)), ceil_div(N, int(bN)));
  int smemBytes = sizeof(SMEMStorage);

  // Get kernel pointer
  auto *kernel_ptr =
      &gemm_device<SMEMStorage, decltype(mA), decltype(mB), decltype(mC),
                   decltype(mD), decltype(mma_tiler), decltype(tiled_mma),
                   decltype(cluster_shape), float, float>;

  // Set kernel attributes
  cudaError_t error = cudaFuncSetAttribute(
      kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes);
  if (error != cudaSuccess) {
    return error;
  }

  // Launch kernel
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster,
                                         smemBytes};
  cutlass::Status status = cutlass::launch_kernel_on_cluster(
      params, (void const *)kernel_ptr, mA, mB, mC, mD, mma_tiler, tiled_mma,
      cluster_shape, alpha, beta);

  return (status == cutlass::Status::kSuccess) ? cudaSuccess
                                               : cudaErrorLaunchFailure;
}

#else

cudaError_t launch_sm100_gemm_f16(void *d_A, void *d_B, void *d_C, void *d_D,
                                  int M, int N, int K, float alpha, float beta,
                                  cudaStream_t stream) {
  return cudaErrorNotSupported;
}

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED
