// CPP extension based on the Cute GEMM tutorial kernels
// https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/blackwell/05_mma_tma_epi_sm100.cu
// derived from so including the Nvidia license.
/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cstdio>
#include <iostream>

// Cutlass includes
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

// Include CuTe verification utilities
#include "cute_verification.hpp"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// Helper function to initialize random values (simplified)
template <class Tensor> void initialize_tensor_simple(Tensor &tensor) {
  for (int i = 0; i < cute::size(tensor); ++i) {
    tensor(i) = static_cast<typename Tensor::value_type>((i % 13) * 0.1f);
  }
}

// The shared memory buffers for A, B, C, and D matrices.
template <class TypeA, class TypeB, class TypeC, class TypeD, class ASmemLayout,
          class BSmemLayout, class CSmemLayout, class DSmemLayout>
struct SharedStorage {
  alignas(128) union {
    alignas(128) struct {
      alignas(128) cute::ArrayEngine<TypeA, cute::cosize_v<ASmemLayout>> A;
      alignas(128) cute::ArrayEngine<TypeB, cute::cosize_v<BSmemLayout>> B;
    } mainloop;
    alignas(128) cute::ArrayEngine<TypeC, cute::cosize_v<CSmemLayout>> C;
    alignas(128) cute::ArrayEngine<TypeD, cute::cosize_v<DSmemLayout>> D;
  } tensors;

  alignas(16) cute::uint64_t mma_barrier;
  alignas(16) cute::uint64_t tma_barrier;
  alignas(16) cute::uint32_t tmem_base_ptr;

  CUTE_DEVICE constexpr auto tensor_sA() {
    return make_tensor(make_smem_ptr(tensors.mainloop.A.begin()),
                       ASmemLayout{});
  }
  CUTE_DEVICE constexpr auto tensor_sB() {
    return make_tensor(make_smem_ptr(tensors.mainloop.B.begin()),
                       BSmemLayout{});
  }
  CUTE_DEVICE constexpr auto tensor_sC() {
    return make_tensor(make_smem_ptr(tensors.C.begin()), CSmemLayout{});
  }
  CUTE_DEVICE constexpr auto tensor_sD() {
    return make_tensor(make_smem_ptr(tensors.D.begin()), DSmemLayout{});
  }
};

// The device kernel
template <class SharedStorage, class ATensor, class BTensor, class CTensor,
          class DTensor, class MmaTiler_MNK, class EpiTiler_MN, class TiledMMA,
          class ClusterShape_MNK, class TmaAtomA, class TmaAtomB,
          class TmaAtomC, class TmaAtomD, class Alpha, class Beta>
__global__ static void gemm_device(ATensor mA, BTensor mB, CTensor mC,
                                   DTensor mD, MmaTiler_MNK mma_tiler,
                                   EpiTiler_MN epi_tiler_mn, TiledMMA tiled_mma,
                                   ClusterShape_MNK cluster_shape,
                                   CUTE_GRID_CONSTANT TmaAtomA const tma_atom_A,
                                   CUTE_GRID_CONSTANT TmaAtomB const tma_atom_B,
                                   CUTE_GRID_CONSTANT TmaAtomC const tma_atom_C,
                                   CUTE_GRID_CONSTANT TmaAtomD const tma_atom_D,
                                   Alpha alpha, Beta beta) {
  // Step 1: The Prologue.
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

  extern __shared__ char shared_memory[];
  SharedStorage &shared_storage =
      *reinterpret_cast<SharedStorage *>(shared_memory);

  Tensor tCsA = shared_storage.tensor_sA();
  Tensor tCsB = shared_storage.tensor_sB();

  auto mma_v = get<0>(mma_coord_vmnk);
  ThrMMA cta_mma = tiled_mma.get_slice(mma_v);
  Tensor tCgA = cta_mma.partition_A(gA);
  Tensor tCgB = cta_mma.partition_B(gB);
  Tensor tCgC = cta_mma.partition_C(gC);
  Tensor tCgD = cta_mma.partition_C(gD);

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

  // Step 2: The Mainloop.
  tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

  for (int k_tile = 0; k_tile < size<3>(tCgA); ++k_tile) {
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

  // Step 3: The Epilogue.
  auto epi_tiler_v = make_tile(epi_tiler_mn);
  Tensor tAcc_epi = zipped_divide(tCtAcc, epi_tiler_v);
  Tensor gC_epi = zipped_divide(tCgC, epi_tiler_v);
  Tensor gD_epi = zipped_divide(tCgD, epi_tiler_v);

  Tensor sC_epi = shared_storage.tensor_sC();
  Tensor sD_epi = shared_storage.tensor_sD();

  auto [tGS_gC, tGS_sC] = tma_partition(tma_atom_C, sC_epi, gC_epi);
  auto [tSG_gD, tSG_sD] = tma_partition(tma_atom_D, sD_epi, gD_epi);

  tma_transaction_bytes = sizeof(make_tensor_like(tGS_sC));

  TiledCopy t2r_copy =
      make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tAcc_epi(_, _0{}));
  ThrCopy thr_t2r = t2r_copy.get_slice(threadIdx.x);
  Tensor tTR_tAcc = thr_t2r.partition_S(tAcc_epi);
  Tensor tTR_sC = thr_t2r.partition_D(sC_epi);
  Tensor tTR_sD = thr_t2r.partition_D(sD_epi);
  Tensor tTR_rC = make_tensor_like(tTR_sC);
  Tensor tTR_rD = make_fragment_like(tTR_sD);

  CUTE_UNROLL
  for (int epi_tile_idx = 0; epi_tile_idx < size<2>(tTR_tAcc); ++epi_tile_idx) {
    if (elect_one_warp && elect_one_thr) {
      cute::set_barrier_transaction_bytes(shared_storage.tma_barrier,
                                          tma_transaction_bytes);
      copy(tma_atom_C.with(shared_storage.tma_barrier, 0),
           tGS_gC(_, epi_tile_idx), tGS_sC);
    }
    cute::wait_barrier(shared_storage.tma_barrier, tma_barrier_phase_bit);
    tma_barrier_phase_bit ^= 1;

    copy_aligned(tTR_sC, tTR_rC);
    copy(t2r_copy, tTR_tAcc(_, _, epi_tile_idx), tTR_rD);
    axpby(beta, tTR_rC, alpha, tTR_rD);

    __syncthreads();
    copy_aligned(tTR_rD, tTR_sD);

    tma_store_fence();
    __syncthreads();
    if (elect_one_warp && elect_one_thr) {
      copy(tma_atom_D, tSG_sD, tSG_gD(_, epi_tile_idx));
      tma_store_arrive();
      tma_store_wait<0>();
    }
    __syncthreads();
  }
  __syncthreads();

  if (elect_one_warp) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(shared_storage.tmem_base_ptr,
                        TmemAllocator::Sm100TmemCapacityColumns);
  }
}

template <class TypeA, class LayoutA, class TypeB, class LayoutB, class TypeC,
          class LayoutC, class TypeD, class LayoutD, class Alpha, class Beta>
void gemm_host_f16xf16_f32_f32_tnt(TypeA const *device_ptr_A, LayoutA layout_A,
                                   TypeB const *device_ptr_B, LayoutB layout_B,
                                   TypeC const *device_ptr_C, LayoutC layout_C,
                                   TypeD *device_ptr_D, LayoutD layout_D,
                                   Alpha const alpha, Beta const beta) {
  assert(shape<0>(layout_A) == shape<0>(layout_C));
  assert(shape<0>(layout_A) == shape<0>(layout_D));
  assert(shape<0>(layout_B) == shape<1>(layout_C));
  assert(shape<0>(layout_B) == shape<1>(layout_D));
  assert(shape<1>(layout_A) == shape<1>(layout_B));

  Tensor mA = make_tensor(make_gmem_ptr(device_ptr_A), layout_A);
  Tensor mB = make_tensor(make_gmem_ptr(device_ptr_B), layout_B);
  Tensor mC = make_tensor(make_gmem_ptr(device_ptr_C), layout_C);
  Tensor mD = make_tensor(make_gmem_ptr(device_ptr_D), layout_D);

  auto Gemm_M = shape<0>(layout_A);
  auto Gemm_N = shape<0>(layout_B);
  auto Gemm_K = shape<1>(layout_A);

  TiledMMA tiled_mma = make_tiled_mma(
      SM100_MMA_F16BF16_2x1SM_SS<TypeA, TypeB, TypeC, 256, 256, UMMA::Major::K,
                                 UMMA::Major::K>{});

  auto bM = tile_size<0>(tiled_mma);
  auto bN = tile_size<1>(tiled_mma);
  auto bK = tile_size<2>(tiled_mma) * Int<4>{};
  auto mma_tiler = make_shape(bM, bN, bK);

  if (not evenly_divides(shape(mma_tiler), tile_shape(tiled_mma))) {
    throw std::runtime_error(
        "The MMA Shape should evenly divide the MMA Tiler.");
  }

  if (not evenly_divides(make_shape(Gemm_M, Gemm_N, Gemm_K), mma_tiler)) {
    throw std::runtime_error("OOB accesses are not supported. MmaTiler_MNK "
                             "should evenly divide ProblemShape_MNK.");
  }

  auto mma_shape_A = partition_shape_A(
      tiled_mma, make_shape(size<0>(mma_tiler), size<2>(mma_tiler)));
  auto mma_shape_B = partition_shape_B(
      tiled_mma, make_shape(size<1>(mma_tiler), size<2>(mma_tiler)));

  auto sA_layout =
      UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TypeA>{}, mma_shape_A);
  auto sB_layout =
      UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TypeB>{}, mma_shape_B);

  auto mma_shape_C = partition_shape_C(
      tiled_mma, make_shape(size<0>(mma_tiler), size<1>(mma_tiler)));
  auto epi_tiler =
      make_tile(size<0, 0>(mma_shape_C), size<0, 1>(mma_shape_C) / Int<4>{});

  auto sC_layout_mn =
      tile_to_shape(UMMA::Layout_K_SW128_Atom<TypeC>{},
                    make_shape(size<0>(epi_tiler), size<1>(epi_tiler)));
  auto sC_layout = group<0, 2>(sC_layout_mn);

  auto sD_layout_mn =
      tile_to_shape(UMMA::Layout_K_SW128_Atom<TypeD>{},
                    make_shape(size<0>(epi_tiler), size<1>(epi_tiler)));
  auto sD_layout = group<0, 2>(sD_layout_mn);

  using SMEMStorage = SharedStorage<TypeA, TypeB, TypeC, TypeD,
                                    decltype(sA_layout), decltype(sB_layout),
                                    decltype(sC_layout), decltype(sD_layout)>;

  auto cluster_shape = make_shape(Int<4>{}, Int<4>{}, Int<1>{});
  Layout cluster_layout_vmnk =
      tiled_divide(make_layout(cluster_shape),
                   make_tile(typename decltype(tiled_mma)::AtomThrID{}));

  Copy_Atom tma_atom_A =
      make_tma_atom_A_sm100(SM100_TMA_2SM_LOAD_MULTICAST{}, mA, sA_layout,
                            mma_tiler, tiled_mma, cluster_layout_vmnk);
  Tensor mA_tma = tma_atom_A.get_tma_tensor(shape(mA));

  Copy_Atom tma_atom_B =
      make_tma_atom_B_sm100(SM100_TMA_2SM_LOAD_MULTICAST{}, mB, sB_layout,
                            mma_tiler, tiled_mma, cluster_layout_vmnk);
  Tensor mB_tma = tma_atom_B.get_tma_tensor(shape(mB));

  Copy_Atom tma_atom_C =
      make_tma_atom(SM90_TMA_LOAD{}, mC, sC_layout, epi_tiler);
  Tensor mC_tma = tma_atom_C.get_tma_tensor(shape(mC));

  Copy_Atom tma_atom_D =
      make_tma_atom(SM90_TMA_STORE{}, mD, sD_layout, epi_tiler);
  Tensor mD_tma = tma_atom_D.get_tma_tensor(shape(mD));

  dim3 dimBlock(128);
  dim3 dimCluster(size<0>(cluster_shape), size<1>(cluster_shape),
                  size<2>(cluster_shape));
  dim3 dimGrid(round_up(size(ceil_div(Gemm_M, bM)), dimCluster.x),
               round_up(size(ceil_div(Gemm_N, bN)), dimCluster.y));
  int smemBytes = sizeof(SMEMStorage);

  auto *kernel_ptr =
      &gemm_device<SMEMStorage, decltype(mA_tma), decltype(mB_tma),
                   decltype(mC_tma), decltype(mD_tma), decltype(mma_tiler),
                   decltype(epi_tiler), decltype(tiled_mma),
                   decltype(cluster_shape), decltype(tma_atom_A),
                   decltype(tma_atom_B), decltype(tma_atom_C),
                   decltype(tma_atom_D), Alpha, Beta>;

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
      kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes));

  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster,
                                         smemBytes};
  cutlass::Status status = cutlass::launch_kernel_on_cluster(
      params, (void const *)kernel_ptr, mA_tma, mB_tma, mC_tma, mD_tma,
      mma_tiler, epi_tiler, tiled_mma, cluster_shape, tma_atom_A, tma_atom_B,
      tma_atom_C, tma_atom_D, alpha, beta);
  CUTE_CHECK_LAST();

  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("CUTLASS kernel launch failed");
  }
}

#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// PyTorch interface function for verification
torch::Tensor cute_blackwell_gemm_with_verification_cuda(
    const torch::Tensor &A, const torch::Tensor &B, const torch::Tensor &C,
    float alpha, float beta, bool verify, bool verbose) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  // Check input tensors
  TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
  TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
  TORCH_CHECK(C.is_cuda(), "C must be a CUDA tensor");
  TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
  TORCH_CHECK(B.dtype() == torch::kFloat16, "B must be float16");
  TORCH_CHECK(C.dtype() == torch::kFloat32, "C must be float32");
  TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
  TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
  TORCH_CHECK(C.is_contiguous(), "C must be contiguous");

  auto M = A.size(0);
  auto K = A.size(1);
  auto N = B.size(0);

  TORCH_CHECK(B.size(1) == K, "Dimension mismatch: A.size(1) != B.size(1)");
  TORCH_CHECK(C.size(0) == M, "Dimension mismatch: C.size(0) != A.size(0)");
  TORCH_CHECK(C.size(1) == N, "Dimension mismatch: C.size(1) != B.size(0)");

  // Create output tensor
  auto D = torch::empty(
      {M, N}, torch::TensorOptions().dtype(torch::kFloat32).device(A.device()));

  // Set up CuTe layouts
  using TypeA = cutlass::half_t;
  using TypeB = cutlass::half_t;
  using TypeC = float;
  using TypeD = float;

  // A tensor MxK K-major (Layout T = Row-Major)
  Layout layout_A = make_layout(make_shape(M, K), make_stride(K, Int<1>{}));
  // B tensor NxK K-major (Layout N = Column-Major)
  Layout layout_B = make_layout(make_shape(N, K), make_stride(K, Int<1>{}));
  // C tensor MxN N-major (Layout T = Row-Major)
  Layout layout_C = make_layout(make_shape(M, N), make_stride(N, Int<1>{}));
  // D tensor MxN N-major (Layout T = Row-Major)
  Layout layout_D = make_layout(make_shape(M, N), make_stride(N, Int<1>{}));

  // Launch kernel
  gemm_host_f16xf16_f32_f32_tnt(
      reinterpret_cast<TypeA const *>(A.data_ptr()), layout_A,
      reinterpret_cast<TypeB const *>(B.data_ptr()), layout_B,
      reinterpret_cast<TypeC const *>(C.data_ptr()), layout_C,
      reinterpret_cast<TypeD *>(D.data_ptr()), layout_D, alpha, beta);

  // Verification if requested
  if (verify) {
    // Copy tensors to host for verification
    auto A_host = A.cpu();
    auto B_host = B.cpu();
    auto C_host = C.cpu();
    auto D_host = D.cpu();

    // Create CuTe tensors for host verification
    auto host_tensor_A =
        make_tensor(reinterpret_cast<TypeA *>(A_host.data_ptr()), layout_A);
    auto host_tensor_B =
        make_tensor(reinterpret_cast<TypeB *>(B_host.data_ptr()), layout_B);
    auto host_tensor_C =
        make_tensor(reinterpret_cast<TypeC *>(C_host.data_ptr()), layout_C);
    auto host_tensor_D =
        make_tensor(reinterpret_cast<TypeD *>(D_host.data_ptr()), layout_D);

    // Create reference result
    auto ref_D_data = std::vector<TypeD>(M * N);
    auto host_ref_tensor_D = make_tensor(ref_D_data.data(), layout_D);

    // Compute reference
    reference_gemm<float>(host_tensor_A, host_tensor_B, host_tensor_C,
                          host_ref_tensor_D, alpha, beta);

    // Compare results
    bool verification_passed =
        compare_results(host_tensor_A, host_tensor_B, host_tensor_C,
                        host_tensor_D, host_ref_tensor_D,
                        false,  // print_diff
                        verbose // verbose
        );

    if (!verification_passed) {
      std::cerr << "Warning: CuTe GEMM verification failed!" << std::endl;
      if (verbose) {
        // Print detailed comparison on failure
        compare_results(host_tensor_A, host_tensor_B, host_tensor_C,
                        host_tensor_D, host_ref_tensor_D, true, true);
      }
    } else if (verbose) {
      std::cout << "CuTe GEMM verification passed!" << std::endl;
    }
  }

  return D;
#else
  TORCH_CHECK(false, "CuTe Blackwell GEMM requires SM100 support, but "
                     "CUTLASS_ARCH_MMA_SM100_SUPPORTED is not defined");
#endif
}

// PyTorch interface function
torch::Tensor cute_blackwell_gemm_cuda(const torch::Tensor &A,
                                       const torch::Tensor &B,
                                       const torch::Tensor &C, float alpha,
                                       float beta) {
  return cute_blackwell_gemm_with_verification_cuda(A, B, C, alpha, beta, false,
                                                    false);
}
