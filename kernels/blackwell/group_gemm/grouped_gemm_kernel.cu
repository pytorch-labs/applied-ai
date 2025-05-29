#include "grouped_gemm_kernel.h"
#include <iostream>

// CUTLASS includes

// Cutlass includes

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"

#include "cutlass/layout/matrix.h"

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/tensor_ref.h"

// utils

#include "cutlass/util/command_line.h"

#include "cutlass/util/distribution.h"

#include "cutlass/util/host_tensor.h"

#include "cutlass/util/packed_stride.hpp"

#include "cutlass/util/tensor_view_io.h"

#include "cutlass/util/reference/device/gemm.h"

#include "cutlass/util/reference/device/tensor_compare.h"

#include "cutlass/util/reference/device/tensor_fill.h"
using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

namespace grouped_gemm {

// 1SM Configuration
struct MMA1SMConfig {
  using MmaTileShape = Shape<_128, _256, Int<128 / sizeof(ElementA)>>;
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmSm100;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
};

// 2SM Configuration
struct MMA2SMConfig {
  using MmaTileShape = Shape<_256, _256, Int<128 / sizeof(ElementA)>>;
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmSm100;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;
};

template <typename ScheduleConfig> struct GemmBuilder {
  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, typename ScheduleConfig::MmaTileShape,
          ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator, ElementAccumulator, ElementC, LayoutC *,
          AlignmentC, ElementC, LayoutC *, AlignmentC,
          typename ScheduleConfig::EpilogueSchedule,
          cutlass::epilogue::fusion::LinearCombination<
              ElementC, ElementAccumulator>>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementA, LayoutA *, AlignmentA, ElementB,
          LayoutB *, AlignmentB, ElementAccumulator,
          typename ScheduleConfig::MmaTileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          typename ScheduleConfig::KernelSchedule>::CollectiveOp;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop,
                                           CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

using Gemm1SM = typename GemmBuilder<MMA1SMConfig>::Gemm;
using Gemm2SM = typename GemmBuilder<MMA2SMConfig>::Gemm;

template <typename Gemm>
GroupedGemmResult run_grouped_gemm_cuda_impl(const GroupedGemmParams &params) {

  const int num_groups = params.num_groups;

  // Collect problem sizes and validate tensors
  std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host;
  std::vector<ElementA *> ptr_A_host;
  std::vector<ElementB *> ptr_B_host;
  std::vector<ElementC *> ptr_C_host;
  std::vector<ElementC *> ptr_D_host;
  std::vector<StrideA> stride_A_host;
  std::vector<StrideB> stride_B_host;
  std::vector<StrideC> stride_C_host;
  std::vector<StrideD> stride_D_host;

  int64_t total_output_elements = 0;

  for (int i = 0; i < num_groups; ++i) {
    int M = params.problem_sizes[i].M;
    int N = params.problem_sizes[i].N;
    int K = params.problem_sizes[i].K;

    problem_sizes_host.push_back({M, N, K});

    ptr_A_host.push_back(params.A_ptrs[i]);
    ptr_B_host.push_back(params.B_ptrs[i]);
    ptr_C_host.push_back(params.C_ptrs[i]);

    // Create strides
    stride_A_host.push_back(
        cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1}));
    stride_B_host.push_back(
        cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1}));
    stride_C_host.push_back(
        cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1}));
    stride_D_host.push_back(
        cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1}));

    total_output_elements += M * N;
  }

  // Allocate output tensor
  ElementC *output_ptr;
  CUDA_CHECK(cudaMalloc(&output_ptr, total_output_elements * sizeof(ElementC)));

  // Set up output pointers
  int64_t offset = 0;
  for (int i = 0; i < num_groups; ++i) {
    ptr_D_host.push_back(output_ptr + offset);
    int M = std::get<0>(problem_sizes_host[i]);
    int N = std::get<1>(problem_sizes_host[i]);
    offset += M * N;
  }

  // Allocate device memory for problem sizes and pointers
  cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape>
      problem_sizes(num_groups);
  problem_sizes.copy_from_host(problem_sizes_host.data());

  cutlass::DeviceAllocation<const ElementA *> ptr_A(num_groups);
  ptr_A.copy_from_host(ptr_A_host.data());

  cutlass::DeviceAllocation<const ElementB *> ptr_B(num_groups);
  ptr_B.copy_from_host(ptr_B_host.data());

  cutlass::DeviceAllocation<const ElementC *> ptr_C(num_groups);
  ptr_C.copy_from_host(ptr_C_host.data());

  cutlass::DeviceAllocation<ElementC *> ptr_D(num_groups);
  ptr_D.copy_from_host(ptr_D_host.data());

  cutlass::DeviceAllocation<StrideA> stride_A(num_groups);
  stride_A.copy_from_host(stride_A_host.data());

  cutlass::DeviceAllocation<StrideB> stride_B(num_groups);
  stride_B.copy_from_host(stride_B_host.data());

  cutlass::DeviceAllocation<StrideC> stride_C(num_groups);
  stride_C.copy_from_host(stride_C_host.data());

  cutlass::DeviceAllocation<StrideD> stride_D(num_groups);
  stride_D.copy_from_host(stride_D_host.data());

  // Set up hardware info
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);

  if constexpr (!is_static_v<ClusterShape>) {
    hw_info.cluster_shape = dim3(params.use_2sm_config ? 2 : 1, 1, 1);
    hw_info.cluster_shape_fallback = dim3(1, 1, 1);
  }

  // Set up epilogue arguments
  typename Gemm::Arguments::EpilogueArguments epilogue_args;
  epilogue_args.thread.alpha = params.alpha;
  epilogue_args.thread.beta = params.beta;
  epilogue_args.thread.alpha_ptr = nullptr;
  epilogue_args.thread.beta_ptr = nullptr;
  epilogue_args.thread.alpha_ptr_array = nullptr;
  epilogue_args.thread.beta_ptr_array = nullptr;
  epilogue_args.thread.dAlpha = {_0{}, _0{}, 0};
  epilogue_args.thread.dBeta = {_0{}, _0{}, 0};
  epilogue_args.ptr_C = ptr_C.get();
  epilogue_args.dC = stride_C.get();
  epilogue_args.ptr_D = ptr_D.get();
  epilogue_args.dD = stride_D.get();

  // Set up scheduler arguments
  typename Gemm::GemmKernel::TileSchedulerArguments scheduler_args;
  if (params.raster_order == RasterOrder::AlongN) {
    scheduler_args.raster_order =
        cutlass::gemm::kernel::detail::PersistentTileSchedulerSm100GroupParams<
            typename ProblemShape::UnderlyingProblemShape>::RasterOrderOptions::
            AlongN;
  } else {
    scheduler_args.raster_order =
        cutlass::gemm::kernel::detail::PersistentTileSchedulerSm100GroupParams<
            typename ProblemShape::UnderlyingProblemShape>::RasterOrderOptions::
            AlongM;
  }

  // Create GEMM arguments
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_groups, problem_sizes.get(), nullptr},
      {ptr_A.get(), stride_A.get(), ptr_B.get(), stride_B.get()},
      epilogue_args,
      hw_info,
      scheduler_args};

  // Create and run GEMM
  Gemm gemm;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  auto status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    CUDA_CHECK(cudaFree(output_ptr));
    throw std::runtime_error(
        "CUTLASS kernel cannot implement the given arguments");
  }

  status = gemm.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    CUDA_CHECK(cudaFree(output_ptr));
    throw std::runtime_error("CUTLASS kernel initialization failed");
  }

  status = gemm.run();
  if (status != cutlass::Status::kSuccess) {
    CUDA_CHECK(cudaFree(output_ptr));
    throw std::runtime_error("CUTLASS kernel execution failed");
  }

  // Wait for completion
  CUDA_CHECK(cudaDeviceSynchronize());

  GroupedGemmResult result;
  result.output_ptr = output_ptr;
  result.total_elements = total_output_elements;
  result.success = true;

  return result;
}

// Template instantiation
GroupedGemmResult run_grouped_gemm_1sm(const GroupedGemmParams &params) {
  return run_grouped_gemm_cuda_impl<Gemm1SM>(params);
}

GroupedGemmResult run_grouped_gemm_2sm(const GroupedGemmParams &params) {
  return run_grouped_gemm_cuda_impl<Gemm2SM>(params);
}

void free_output_tensor(ElementC *ptr) {
  if (ptr) {
    CUDA_CHECK(cudaFree(ptr));
  }
}

} // namespace grouped_gemm

#else // !defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

namespace grouped_gemm {

GroupedGemmResult run_grouped_gemm_1sm(const GroupedGemmParams &params) {
  throw std::runtime_error("CUTLASS SM100 support not available. Please "
                           "compile with proper CUDA and CUTLASS versions.");
}

GroupedGemmResult run_grouped_gemm_2sm(const GroupedGemmParams &params) {
  throw std::runtime_error("CUTLASS SM100 support not available. Please "
                           "compile with proper CUDA and CUTLASS versions.");
}

void free_output_tensor(ElementC *ptr) {
  // No-op
}

} // namespace grouped_gemm

#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
