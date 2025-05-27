/***************************************************************************************************
 * PyTorch C++ Extension for Blackwell SM100 GEMM
 * Based on CuTe Tutorial 04: tcgen05.mma (UMMA)
 **************************************************************************************************/

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdio>
#include <iostream>

// Cutlass includes (must come first)
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/half.h>
#include <cutlass/util/print_error.hpp>

// CuTe includes
#include <cute/algorithm/cooperative_copy.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/tensor.hpp>

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// SharedStorage template
template <class TypeA, class TypeB, class ASmemLayout, class BSmemLayout>
struct SharedStorage {
  alignas(128) cute::ArrayEngine<TypeA, cute::cosize_v<ASmemLayout>> A;
  alignas(128) cute::ArrayEngine<TypeB, cute::cosize_v<BSmemLayout>> B;

  alignas(16) cute::uint64_t mma_barrier;
  alignas(16) cute::uint64_t tma_barrier;
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
          class ClusterShape_MNK, class TmaAtomA, class TmaAtomB, class Alpha,
          class Beta>
__global__ static void gemm_device(ATensor mA, BTensor mB, CTensor mC,
                                   DTensor mD, MmaTiler_MNK mma_tiler,
                                   TiledMMA tiled_mma,
                                   ClusterShape_MNK cluster_shape,
                                   CUTE_GRID_CONSTANT TmaAtomA const tma_atom_A,
                                   CUTE_GRID_CONSTANT TmaAtomB const tma_atom_B,
                                   Alpha alpha, Beta beta) {
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

  // SMEM tensors
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

  // MMA Fragment Allocation
  Tensor tCrA = cta_mma.make_fragment_A(tCsA);
  Tensor tCrB = cta_mma.make_fragment_B(tCsB);
  Tensor tCtAcc = cta_mma.make_fragment_C(tCgC);

  uint32_t elect_one_thr = cute::elect_one_sync();
  uint32_t elect_one_warp = (threadIdx.x / 32 == 0);

  using TmemAllocator = cute::TMEM::Allocator2Sm;
  TmemAllocator tmem_allocator{};

  if (elect_one_warp) {
    tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns,
                            &shared_storage.tmem_base_ptr);
  }
  __syncthreads();
  tCtAcc.data() = shared_storage.tmem_base_ptr;

  // TMA Setup
  auto cta_in_cluster_coord_vmnk =
      cluster_layout_vmnk.get_flat_coord(int(cute::block_rank_in_cluster()));
  auto elect_one_cta = get<0>(cta_in_cluster_coord_vmnk) == Int<0>{};

  auto [tAgA, tAsA] =
      tma_partition(tma_atom_A, get<2>(cta_in_cluster_coord_vmnk),
                    make_layout(size<2>(cluster_layout_vmnk)),
                    group_modes<0, 3>(tCsA), group_modes<0, 3>(tCgA));

  auto [tBgB, tBsB] =
      tma_partition(tma_atom_B, get<1>(cta_in_cluster_coord_vmnk),
                    make_layout(size<1>(cluster_layout_vmnk)),
                    group_modes<0, 3>(tCsB), group_modes<0, 3>(tCgB));

  uint16_t tma_mcast_mask_a = create_tma_multicast_mask<2>(
      cluster_layout_vmnk, cta_in_cluster_coord_vmnk);
  uint16_t tma_mcast_mask_b = create_tma_multicast_mask<1>(
      cluster_layout_vmnk, cta_in_cluster_coord_vmnk);
  uint16_t mma_mcast_mask_c =
      create_tma_multicast_mask<0, 1>(cluster_layout_vmnk,
                                      cta_in_cluster_coord_vmnk) |
      create_tma_multicast_mask<0, 2>(cluster_layout_vmnk,
                                      cta_in_cluster_coord_vmnk);

  int tma_transaction_bytes =
      size<0>(cluster_layout_vmnk) * sizeof(make_tensor_like(tAsA)) +
      size<0>(cluster_layout_vmnk) * sizeof(make_tensor_like(tBsB));

  // Barrier Initialization
  if (elect_one_warp && elect_one_thr) {
    int num_mcast_participants =
        size<1>(cluster_layout_vmnk) + size<2>(cluster_layout_vmnk) - 1;
    cute::initialize_barrier(shared_storage.mma_barrier,
                             num_mcast_participants);
    cute::initialize_barrier(shared_storage.tma_barrier, 1);
  }
  int mma_barrier_phase_bit = 0;
  int tma_barrier_phase_bit = 0;
  cute::cluster_sync();

  // Step 2: The Mainloop
  tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

  for (int k_tile = 0; k_tile < size<3>(tCgA); ++k_tile) {
    // Step 2a: Load A and B tiles
    if (elect_one_warp && elect_one_thr) {
      if (elect_one_cta) {
        cute::set_barrier_transaction_bytes(shared_storage.tma_barrier,
                                            tma_transaction_bytes);
      }
      copy(tma_atom_A.with(shared_storage.tma_barrier, tma_mcast_mask_a),
           tAgA(_, k_tile), tAsA);
      copy(tma_atom_B.with(shared_storage.tma_barrier, tma_mcast_mask_b),
           tBgB(_, k_tile), tBsB);
    }

    // Step 2b: Execute the MMAs for this tile
    if (elect_one_cta) {
      cute::wait_barrier(shared_storage.tma_barrier, tma_barrier_phase_bit);
      tma_barrier_phase_bit ^= 1;

      if (elect_one_warp) {
        for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
          gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCtAcc);
          tiled_mma.accumulate_ = UMMA::ScaleOut::One;
        }
        cutlass::arch::umma_arrive_multicast_2x1SM(&shared_storage.mma_barrier,
                                                   mma_mcast_mask_c);
      }
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

  // AXPBY: tDrC = alpha * tDrAcc + beta * tDrC
  axpby(alpha, tDrAcc, beta, tDrC);
  copy(tDrC, tDgD);

  __syncthreads();

  if (elect_one_warp) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(shared_storage.tmem_base_ptr,
                        TmemAllocator::Sm100TmemCapacityColumns);
  }
}

// Host function template
template <class TypeA, class LayoutA, class TypeB, class LayoutB, class TypeC,
          class LayoutC, class TypeD, class LayoutD, class Alpha, class Beta>
void gemm_host_f16xf16_f32_f32_tnt(TypeA const *device_ptr_A, LayoutA layout_A,
                                   TypeB const *device_ptr_B, LayoutB layout_B,
                                   TypeC const *device_ptr_C, LayoutC layout_C,
                                   TypeD *device_ptr_D, LayoutD layout_D,
                                   Alpha const alpha, Beta const beta) {
  // Represent the full tensors in global memory
  Tensor mA = make_tensor(make_gmem_ptr(device_ptr_A), layout_A);
  Tensor mB = make_tensor(make_gmem_ptr(device_ptr_B), layout_B);
  Tensor mC = make_tensor(make_gmem_ptr(device_ptr_C), layout_C);
  Tensor mD = make_tensor(make_gmem_ptr(device_ptr_D), layout_D);

  // Create TiledMma
  TiledMMA tiled_mma = make_tiled_mma(
      SM100_MMA_F16BF16_2x1SM_SS<TypeA, TypeB, TypeC, 256, 256, UMMA::Major::K,
                                 UMMA::Major::K>{});

  // Define MMA tiler sizes
  auto bM = tile_size<0>(tiled_mma);
  auto bN = tile_size<1>(tiled_mma);
  auto bK = tile_size<2>(tiled_mma) * Int<4>{};
  auto mma_tiler = make_shape(bM, bN, bK);

  // Determine SMEM layouts
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

  // Cluster setup
  auto cluster_shape = make_shape(Int<4>{}, Int<4>{}, Int<1>{});
  Layout cluster_layout_vmnk =
      tiled_divide(make_layout(cluster_shape),
                   make_tile(typename decltype(tiled_mma)::AtomThrID{}));

  // TMA atoms
  Copy_Atom tma_atom_A =
      make_tma_atom_A_sm100(SM100_TMA_2SM_LOAD_MULTICAST{}, mA, sA_layout,
                            mma_tiler, tiled_mma, cluster_layout_vmnk);

  Copy_Atom tma_atom_B =
      make_tma_atom_B_sm100(SM100_TMA_2SM_LOAD_MULTICAST{}, mB, sB_layout,
                            mma_tiler, tiled_mma, cluster_layout_vmnk);

  Tensor mA_tma = tma_atom_A.get_tma_tensor(shape(mA));
  Tensor mB_tma = tma_atom_B.get_tma_tensor(shape(mB));

  // Launch kernel
  auto Gemm_M = shape<0>(layout_A);
  auto Gemm_N = shape<0>(layout_B);

  dim3 dimBlock(128);
  dim3 dimCluster(size<0>(cluster_shape), size<1>(cluster_shape),
                  size<2>(cluster_shape));
  dim3 dimGrid(round_up(size(ceil_div(Gemm_M, bM)), dimCluster.x),
               round_up(size(ceil_div(Gemm_N, bN)), dimCluster.y));
  int smemBytes = sizeof(SMEMStorage);

  auto *kernel_ptr =
      &gemm_device<SMEMStorage, decltype(mA_tma), decltype(mB_tma),
                   decltype(mC), decltype(mD), decltype(mma_tiler),
                   decltype(tiled_mma), decltype(cluster_shape),
                   decltype(tma_atom_A), decltype(tma_atom_B), Alpha, Beta>;

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
      kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes));

  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster,
                                         smemBytes};
  cutlass::Status status = cutlass::launch_kernel_on_cluster(
      params, (void const *)kernel_ptr, mA_tma, mB_tma, mC, mD, mma_tiler,
      tiled_mma, cluster_shape, tma_atom_A, tma_atom_B, alpha, beta);
  CUTE_CHECK_LAST();

  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("GEMM kernel launch failed");
  }
}

#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// PyTorch interface function
torch::Tensor blackwell_gemm_f16(const torch::Tensor &A, // [M, K] - row major
                                 const torch::Tensor &B, // [N, K] - row major
                                 const torch::Tensor &C, // [M, N] - row major
                                 float alpha = 1.0f, float beta = 0.0f) {

  // Check inputs
  TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
  TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
  TORCH_CHECK(C.device().is_cuda(), "C must be a CUDA tensor");
  TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
  TORCH_CHECK(B.dtype() == torch::kFloat16, "B must be float16");
  TORCH_CHECK(C.dtype() == torch::kFloat32, "C must be float32");
  TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
  TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
  TORCH_CHECK(C.is_contiguous(), "C must be contiguous");

  const int M = A.size(0);
  const int K = A.size(1);
  const int N = B.size(0);

  TORCH_CHECK(B.size(1) == K, "B.size(1) must equal A.size(1)");
  TORCH_CHECK(C.size(0) == M, "C.size(0) must equal A.size(0)");
  TORCH_CHECK(C.size(1) == N, "C.size(1) must equal B.size(0)");

  // Check problem size constraints for SM100
  TORCH_CHECK(M % 256 == 0, "M must be divisible by 256 for SM100 GEMM");
  TORCH_CHECK(N % 256 == 0, "N must be divisible by 256 for SM100 GEMM");
  TORCH_CHECK(K % 64 == 0, "K must be divisible by 64 for SM100 GEMM");

  // Create output tensor
  torch::Tensor D = torch::empty_like(C);

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

  // Set CUDA device
  at::cuda::CUDAGuard device_guard(A.device());

  // Define data types
  using TypeA = cutlass::half_t;
  using TypeB = cutlass::half_t;
  using TypeC = float;
  using TypeD = float;

  // Create layouts - the tutorial expects K-major for A and B, N-major for C
  // and D A: (M, K) with K-major layout (transpose of row-major)
  auto layout_A = make_layout(make_shape(M, K), make_stride(K, Int<1>{}));
  // B: (N, K) with K-major layout (transpose of row-major)
  auto layout_B = make_layout(make_shape(N, K), make_stride(K, Int<1>{}));
  // C and D: (M, N) with N-major layout (row-major)
  auto layout_C = make_layout(make_shape(M, N), make_stride(N, Int<1>{}));
  auto layout_D = make_layout(make_shape(M, N), make_stride(N, Int<1>{}));

  // Get raw pointers
  TypeA *ptr_A = reinterpret_cast<TypeA *>(A.data_ptr<at::Half>());
  TypeB *ptr_B = reinterpret_cast<TypeB *>(B.data_ptr<at::Half>());
  TypeC *ptr_C = reinterpret_cast<TypeC *>(C.data_ptr<float>());
  TypeD *ptr_D = reinterpret_cast<TypeD *>(D.data_ptr<float>());

  // Call the GEMM function
  gemm_host_f16xf16_f32_f32_tnt(ptr_A, layout_A, ptr_B, layout_B, ptr_C,
                                layout_C, ptr_D, layout_D, alpha, beta);

#else
  throw std::runtime_error(
      "SM100 support not compiled. Need CUTLASS_ARCH_MMA_SM100_SUPPORTED.");
#endif

  return D;
}
