#include "grouped_gemm_kernel.h"
#include <cstring>

// CUTLASS includes
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

// CUTLASS type definitions
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using ElementC = cutlass::half_t;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::ColumnMajor;

constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using ClusterShape = Shape<int32_t, int32_t, _1>;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// 1SM Configuration
using MmaTileShape1SM = Shape<_128, _256, Int<128 / sizeof(ElementA)>>;
using KernelSchedule1SM =
    cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmSm100;
using EpilogueSchedule1SM = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;

// 2SM Configuration
using MmaTileShape2SM = Shape<_256, _256, Int<128 / sizeof(ElementA)>>;
using KernelSchedule2SM =
    cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmSm100;
using EpilogueSchedule2SM = cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;

// Build 1SM GEMM
using CollectiveEpilogue1SM =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, MmaTileShape1SM, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
        ElementAccumulator, ElementC, LayoutC *, AlignmentC, ElementC,
        LayoutC *, AlignmentC, EpilogueSchedule1SM,
        cutlass::epilogue::fusion::LinearCombination<
            ElementC, ElementAccumulator>>::CollectiveOp;

using CollectiveMainloop1SM =
    typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass, ElementA, LayoutA *, AlignmentA, ElementB,
        LayoutB *, AlignmentB, ElementAccumulator, MmaTileShape1SM,
        ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue1SM::SharedStorage))>,
        KernelSchedule1SM>::CollectiveOp;

using GemmKernel1SM =
    cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop1SM,
                                         CollectiveEpilogue1SM>;
using Gemm1SM = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel1SM>;

// Build 2SM GEMM
using CollectiveEpilogue2SM =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, MmaTileShape2SM, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
        ElementAccumulator, ElementC, LayoutC *, AlignmentC, ElementC,
        LayoutC *, AlignmentC, EpilogueSchedule2SM,
        cutlass::epilogue::fusion::LinearCombination<
            ElementC, ElementAccumulator>>::CollectiveOp;

using CollectiveMainloop2SM =
    typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass, ElementA, LayoutA *, AlignmentA, ElementB,
        LayoutB *, AlignmentB, ElementAccumulator, MmaTileShape2SM,
        ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue2SM::SharedStorage))>,
        KernelSchedule2SM>::CollectiveOp;

using GemmKernel2SM =
    cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop2SM,
                                         CollectiveEpilogue2SM>;
using Gemm2SM = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel2SM>;

template <typename Gemm>
GemmResult run_gemm_impl(void **A_ptrs, void **B_ptrs, void **C_ptrs,
                         int *M_sizes, int *N_sizes, int *K_sizes,
                         int num_groups, float alpha, float beta,
                         bool use_2sm_config, bool raster_along_n) {

  GemmResult result = {};

  try {
    // Prepare problem sizes
    std::vector<typename ProblemShape::UnderlyingProblemShape>
        problem_sizes_host;
    int64_t total_elements = 0;

    for (int i = 0; i < num_groups; ++i) {
      problem_sizes_host.push_back({M_sizes[i], N_sizes[i], K_sizes[i]});
      total_elements += M_sizes[i] * N_sizes[i];
    }

    // Allocate output
    ElementC *output_ptr;
    cudaError_t error =
        cudaMalloc(&output_ptr, total_elements * sizeof(ElementC));
    if (error != cudaSuccess) {
      strcpy(result.error_message, "Failed to allocate output");
      return result;
    }

    // Prepare output pointers
    std::vector<ElementC *> D_ptrs_host;
    int64_t offset = 0;
    for (int i = 0; i < num_groups; ++i) {
      D_ptrs_host.push_back(output_ptr + offset);
      offset += M_sizes[i] * N_sizes[i];
    }

    // Prepare strides
    using StrideA = typename Gemm::GemmKernel::InternalStrideA;
    using StrideB = typename Gemm::GemmKernel::InternalStrideB;
    using StrideC = typename Gemm::GemmKernel::InternalStrideC;
    using StrideD = typename Gemm::GemmKernel::InternalStrideD;

    std::vector<StrideA> stride_A_host;
    std::vector<StrideB> stride_B_host;
    std::vector<StrideC> stride_C_host;
    std::vector<StrideD> stride_D_host;

    for (int i = 0; i < num_groups; ++i) {
      stride_A_host.push_back(cutlass::make_cute_packed_stride(
          StrideA{}, {M_sizes[i], K_sizes[i], 1}));
      stride_B_host.push_back(cutlass::make_cute_packed_stride(
          StrideB{}, {N_sizes[i], K_sizes[i], 1}));
      stride_C_host.push_back(cutlass::make_cute_packed_stride(
          StrideC{}, {M_sizes[i], N_sizes[i], 1}));
      stride_D_host.push_back(cutlass::make_cute_packed_stride(
          StrideD{}, {M_sizes[i], N_sizes[i], 1}));
    }

    // Device allocations
    cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape>
        problem_sizes_device(num_groups);
    problem_sizes_device.copy_from_host(problem_sizes_host.data());

    cutlass::DeviceAllocation<const ElementA *> ptr_A_device(num_groups);
    ptr_A_device.copy_from_host(reinterpret_cast<const ElementA **>(A_ptrs));

    cutlass::DeviceAllocation<const ElementB *> ptr_B_device(num_groups);
    ptr_B_device.copy_from_host(reinterpret_cast<const ElementB **>(B_ptrs));

    cutlass::DeviceAllocation<const ElementC *> ptr_C_device(num_groups);
    ptr_C_device.copy_from_host(reinterpret_cast<const ElementC **>(C_ptrs));

    cutlass::DeviceAllocation<ElementC *> ptr_D_device(num_groups);
    ptr_D_device.copy_from_host(D_ptrs_host.data());

    cutlass::DeviceAllocation<StrideA> stride_A_device(num_groups);
    stride_A_device.copy_from_host(stride_A_host.data());

    cutlass::DeviceAllocation<StrideB> stride_B_device(num_groups);
    stride_B_device.copy_from_host(stride_B_host.data());

    cutlass::DeviceAllocation<StrideC> stride_C_device(num_groups);
    stride_C_device.copy_from_host(stride_C_host.data());

    cutlass::DeviceAllocation<StrideD> stride_D_device(num_groups);
    stride_D_device.copy_from_host(stride_D_host.data());

    // Hardware info
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
            hw_info.device_id);

    if constexpr (!is_static_v<ClusterShape>) {
      hw_info.cluster_shape = dim3(use_2sm_config ? 2 : 1, 1, 1);
      hw_info.cluster_shape_fallback = dim3(1, 1, 1);
    }

    // Epilogue arguments
    typename Gemm::Arguments::EpilogueArguments epilogue_args;
    epilogue_args.thread.alpha = alpha;
    epilogue_args.thread.beta = beta;
    epilogue_args.thread.alpha_ptr = nullptr;
    epilogue_args.thread.beta_ptr = nullptr;
    epilogue_args.thread.alpha_ptr_array = nullptr;
    epilogue_args.thread.beta_ptr_array = nullptr;
    epilogue_args.thread.dAlpha = {_0{}, _0{}, 0};
    epilogue_args.thread.dBeta = {_0{}, _0{}, 0};
    epilogue_args.ptr_C = ptr_C_device.get();
    epilogue_args.dC = stride_C_device.get();
    epilogue_args.ptr_D = ptr_D_device.get();
    epilogue_args.dD = stride_D_device.get();

    // Scheduler arguments
    typename Gemm::GemmKernel::TileSchedulerArguments scheduler_args;
    if (raster_along_n) {
      scheduler_args.raster_order = cutlass::gemm::kernel::detail::
          PersistentTileSchedulerSm100GroupParams<
              typename ProblemShape::UnderlyingProblemShape>::
              RasterOrderOptions::AlongN;
    } else {
      scheduler_args.raster_order = cutlass::gemm::kernel::detail::
          PersistentTileSchedulerSm100GroupParams<
              typename ProblemShape::UnderlyingProblemShape>::
              RasterOrderOptions::AlongM;
    }

    // Create GEMM arguments
    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {num_groups, problem_sizes_device.get(), nullptr},
        {ptr_A_device.get(), stride_A_device.get(), ptr_B_device.get(),
         stride_B_device.get()},
        epilogue_args,
        hw_info,
        scheduler_args};

    // Run GEMM
    Gemm gemm;
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    auto status = gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      cudaFree(output_ptr);
      strcpy(result.error_message, "CUTLASS cannot implement arguments");
      return result;
    }

    status = gemm.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
      cudaFree(output_ptr);
      strcpy(result.error_message, "CUTLASS initialization failed");
      return result;
    }

    status = gemm.run();
    if (status != cutlass::Status::kSuccess) {
      cudaFree(output_ptr);
      strcpy(result.error_message, "CUTLASS execution failed");
      return result;
    }

    cudaDeviceSynchronize();

    result.output_ptr = output_ptr;
    result.total_elements = total_elements;
    result.success = true;
    strcpy(result.error_message, "Success");

  } catch (const std::exception &e) {
    strcpy(result.error_message, e.what());
  } catch (...) {
    strcpy(result.error_message, "Unknown error");
  }

  return result;
}

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED

extern "C" {

GemmResult run_grouped_gemm_cuda(void **A_ptrs, void **B_ptrs, void **C_ptrs,
                                 int *M_sizes, int *N_sizes, int *K_sizes,
                                 int num_groups, float alpha, float beta,
                                 bool use_2sm_config, bool raster_along_n) {

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  if (use_2sm_config) {
    return run_gemm_impl<Gemm2SM>(A_ptrs, B_ptrs, C_ptrs, M_sizes, N_sizes,
                                  K_sizes, num_groups, alpha, beta,
                                  use_2sm_config, raster_along_n);
  } else {
    return run_gemm_impl<Gemm1SM>(A_ptrs, B_ptrs, C_ptrs, M_sizes, N_sizes,
                                  K_sizes, num_groups, alpha, beta,
                                  use_2sm_config, raster_along_n);
  }
#else
  GemmResult result = {};
  strcpy(result.error_message, "CUTLASS SM100 support not available");
  return result;
#endif
}

void free_gemm_output(void *ptr) {
  if (ptr) {
    cudaFree(ptr);
  }
}

bool check_requirements(void) {
  // Check CUDA version
  if (__CUDACC_VER_MAJOR__ < 12 ||
      (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 8)) {
    return false;
  }

  // Check GPU architecture
  cudaDeviceProp props;
  int device_id;
  if (cudaGetDevice(&device_id) != cudaSuccess)
    return false;
  if (cudaGetDeviceProperties(&props, device_id) != cudaSuccess)
    return false;

  return (props.major == 10 && props.minor == 0);
}

} // extern "C"
#include "grouped_gemm_kernel.h"
#include <cstring>
#include <iostream>
#include <string>

// CUTLASS includes - only in .cu file
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

// CUTLASS type definitions (only in .cu file)
using ProblemShapeCutlass =
    cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
using ElementACutlass = cutlass::float_e4m3_t;
using ElementBCutlass = cutlass::float_e4m3_t;
using ElementCCutlass = cutlass::half_t;
using ElementAccumulator = float;

// Layout configurations
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::ColumnMajor;

constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementACutlass>::value;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementBCutlass>::value;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementCCutlass>::value;

// Architecture and operator configuration
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using ClusterShape = Shape<int32_t, int32_t, _1>;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// 1SM Configuration
struct MMA1SMConfig {
  using MmaTileShape = Shape<_128, _256, Int<128 / sizeof(ElementACutlass)>>;
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmSm100;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
};

// 2SM Configuration
struct MMA2SMConfig {
  using MmaTileShape = Shape<_256, _256, Int<128 / sizeof(ElementACutlass)>>;
  using KernelSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmSm100;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;
};

template <typename ScheduleConfig> struct GemmBuilder {
  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, typename ScheduleConfig::MmaTileShape,
          ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator, ElementAccumulator, ElementCCutlass, LayoutC *,
          AlignmentC, ElementCCutlass, LayoutC *, AlignmentC,
          typename ScheduleConfig::EpilogueSchedule,
          cutlass::epilogue::fusion::LinearCombination<
              ElementCCutlass, ElementAccumulator>>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementACutlass, LayoutA *, AlignmentA,
          ElementBCutlass, LayoutB *, AlignmentB, ElementAccumulator,
          typename ScheduleConfig::MmaTileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          typename ScheduleConfig::KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      ProblemShapeCutlass, CollectiveMainloop, CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

using Gemm1SM = typename GemmBuilder<MMA1SMConfig>::Gemm;
using Gemm2SM = typename GemmBuilder<MMA2SMConfig>::Gemm;

using StrideA = typename Gemm1SM::GemmKernel::InternalStrideA;
using StrideB = typename Gemm1SM::GemmKernel::InternalStrideB;
using StrideC = typename Gemm1SM::GemmKernel::InternalStrideC;
using StrideD = typename Gemm1SM::GemmKernel::InternalStrideD;

template <typename Gemm>
GroupedGemmResult run_grouped_gemm_cuda_impl(const GroupedGemmParams *params) {

  GroupedGemmResult result = {};
  result.success = false;

  try {
    const int num_groups = params->num_groups;

    // Collect problem sizes and validate tensors
    std::vector<typename ProblemShapeCutlass::UnderlyingProblemShape>
        problem_sizes_host;
    std::vector<ElementACutlass *> ptr_A_host;
    std::vector<ElementBCutlass *> ptr_B_host;
    std::vector<ElementCCutlass *> ptr_C_host;
    std::vector<ElementCCutlass *> ptr_D_host;
    std::vector<StrideA> stride_A_host;
    std::vector<StrideB> stride_B_host;
    std::vector<StrideC> stride_C_host;
    std::vector<StrideD> stride_D_host;

    int64_t total_output_elements = 0;

    for (int i = 0; i < num_groups; ++i) {
      int M = params->problem_sizes[i].M;
      int N = params->problem_sizes[i].N;
      int K = params->problem_sizes[i].K;

      problem_sizes_host.push_back({M, N, K});

      ptr_A_host.push_back(static_cast<ElementACutlass *>(params->A_ptrs[i]));
      ptr_B_host.push_back(static_cast<ElementBCutlass *>(params->B_ptrs[i]));
      ptr_C_host.push_back(static_cast<ElementCCutlass *>(params->C_ptrs[i]));

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
    ElementCCutlass *output_ptr;
    cudaError_t cuda_error = cudaMalloc(
        &output_ptr, total_output_elements * sizeof(ElementCCutlass));
    if (cuda_error != cudaSuccess) {
      strcpy(result.error_message, "Failed to allocate output tensor");
      return result;
    }

    // Set up output pointers
    int64_t offset = 0;
    for (int i = 0; i < num_groups; ++i) {
      ptr_D_host.push_back(output_ptr + offset);
      int M = std::get<0>(problem_sizes_host[i]);
      int N = std::get<1>(problem_sizes_host[i]);
      offset += M * N;
    }

    // Allocate device memory for problem sizes and pointers
    cutlass::DeviceAllocation<
        typename ProblemShapeCutlass::UnderlyingProblemShape>
        problem_sizes(num_groups);
    problem_sizes.copy_from_host(problem_sizes_host.data());

    cutlass::DeviceAllocation<const ElementACutlass *> ptr_A(num_groups);
    ptr_A.copy_from_host(ptr_A_host.data());

    cutlass::DeviceAllocation<const ElementBCutlass *> ptr_B(num_groups);
    ptr_B.copy_from_host(ptr_B_host.data());

    cutlass::DeviceAllocation<const ElementCCutlass *> ptr_C(num_groups);
    ptr_C.copy_from_host(ptr_C_host.data());

    cutlass::DeviceAllocation<ElementCCutlass *> ptr_D(num_groups);
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
      hw_info.cluster_shape = dim3(params->use_2sm_config ? 2 : 1, 1, 1);
      hw_info.cluster_shape_fallback = dim3(1, 1, 1);
    }

    // Set up epilogue arguments
    typename Gemm::Arguments::EpilogueArguments epilogue_args;
    epilogue_args.thread.alpha = params->alpha;
    epilogue_args.thread.beta = params->beta;
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
    if (params->raster_order == RASTER_ORDER_ALONG_N) {
      scheduler_args.raster_order = cutlass::gemm::kernel::detail::
          PersistentTileSchedulerSm100GroupParams<
              typename ProblemShapeCutlass::UnderlyingProblemShape>::
              RasterOrderOptions::AlongN;
    } else {
      scheduler_args.raster_order = cutlass::gemm::kernel::detail::
          PersistentTileSchedulerSm100GroupParams<
              typename ProblemShapeCutlass::UnderlyingProblemShape>::
              RasterOrderOptions::AlongM;
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
      cudaFree(output_ptr);
      strcpy(result.error_message,
             "CUTLASS kernel cannot implement the given arguments");
      return result;
    }

    status = gemm.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
      cudaFree(output_ptr);
      strcpy(result.error_message, "CUTLASS kernel initialization failed");
      return result;
    }

    status = gemm.run();
    if (status != cutlass::Status::kSuccess) {
      cudaFree(output_ptr);
      strcpy(result.error_message, "CUTLASS kernel execution failed");
      return result;
    }

    // Wait for completion
    cuda_error = cudaDeviceSynchronize();
    if (cuda_error != cudaSuccess) {
      cudaFree(output_ptr);
      strcpy(result.error_message, "CUDA synchronization failed");
      return result;
    }

    result.output_ptr = output_ptr;
    result.total_elements = total_output_elements;
    result.success = true;
    strcpy(result.error_message, "Success");

    return result;

  } catch (const std::exception &e) {
    strcpy(result.error_message, e.what());
    return result;
  } catch (...) {
    strcpy(result.error_message, "Unknown error occurred");
    return result;
  }
}

#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

// C interface implementations
extern "C" {

GroupedGemmResult run_grouped_gemm_1sm_c(const GroupedGemmParams *params) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  return run_grouped_gemm_cuda_impl<Gemm1SM>(params);
#else
  GroupedGemmResult result = {};
  result.success = false;
  strcpy(result.error_message, "CUTLASS SM100 support not available");
  return result;
#endif
}

GroupedGemmResult run_grouped_gemm_2sm_c(const GroupedGemmParams *params) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  return run_grouped_gemm_cuda_impl<Gemm2SM>(params);
#else
  GroupedGemmResult result = {};
  result.success = false;
  strcpy(result.error_message, "CUTLASS SM100 support not available");
  return result;
#endif
}

void free_output_tensor_c(void *ptr) {
  if (ptr) {
    cudaFree(ptr);
  }
}

bool check_cuda_version_c(void) {
  return (__CUDACC_VER_MAJOR__ >= 12 &&
          (__CUDACC_VER_MAJOR__ > 12 || __CUDACC_VER_MINOR__ >= 8));
}

bool check_gpu_architecture_c(void) {
  cudaDeviceProp props;
  int current_device_id;
  cudaError_t error;

  error = cudaGetDevice(&current_device_id);
  if (error != cudaSuccess)
    return false;

  error = cudaGetDeviceProperties(&props, current_device_id);
  if (error != cudaSuccess)
    return false;

  return (props.major == 10 && props.minor == 0);
}

} // extern "C"

// C++ wrapper implementations
namespace grouped_gemm {

GroupedGemmResultCpp run_grouped_gemm_1sm(const GroupedGemmParamsCpp &params) {
  // Convert C++ params to C params
  GroupedGemmParams c_params = {};
  c_params.num_groups = params.num_groups;
  c_params.alpha = params.alpha;
  c_params.beta = params.beta;
  c_params.use_2sm_config = params.use_2sm_config;
  c_params.raster_order = (params.raster_order == RasterOrderCpp::AlongN)
                              ? RASTER_ORDER_ALONG_N
                              : RASTER_ORDER_ALONG_M;

  // Allocate and copy problem sizes
  std::vector<ProblemSize> problem_sizes_c(params.num_groups);
  for (int i = 0; i < params.num_groups; ++i) {
    problem_sizes_c[i].M = params.problem_sizes[i].M;
    problem_sizes_c[i].N = params.problem_sizes[i].N;
    problem_sizes_c[i].K = params.problem_sizes[i].K;
  }
  c_params.problem_sizes = problem_sizes_c.data();

  // Convert pointer vectors
  std::vector<void *> A_ptrs_c(params.A_ptrs);
  std::vector<void *> B_ptrs_c(params.B_ptrs);
  std::vector<void *> C_ptrs_c(params.C_ptrs);

  c_params.A_ptrs = A_ptrs_c.data();
  c_params.B_ptrs = B_ptrs_c.data();
  c_params.C_ptrs = C_ptrs_c.data();

  // Call C function
  GroupedGemmResult c_result = run_grouped_gemm_1sm_c(&c_params);

  // Convert result back to C++
  GroupedGemmResultCpp cpp_result;
  cpp_result.output_ptr = c_result.output_ptr;
  cpp_result.total_elements = c_result.total_elements;
  cpp_result.success = c_result.success;
  cpp_result.error_message = std::string(c_result.error_message);

  return cpp_result;
}

GroupedGemmResultCpp run_grouped_gemm_2sm(const GroupedGemmParamsCpp &params) {
  // Similar implementation as 1SM but calls run_grouped_gemm_2sm_c
  GroupedGemmParams c_params = {};
  c_params.num_groups = params.num_groups;
  c_params.alpha = params.alpha;
  c_params.beta = params.beta;
  c_params.use_2sm_config = params.use_2sm_config;
  c_params.raster_order = (params.raster_order == RasterOrderCpp::AlongN)
                              ? RASTER_ORDER_ALONG_N
                              : RASTER_ORDER_ALONG_M;

  std::vector<ProblemSize> problem_sizes_c(params.num_groups);
  for (int i = 0; i < params.num_groups; ++i) {
    problem_sizes_c[i].M = params.problem_sizes[i].M;
    problem_sizes_c[i].N = params.problem_sizes[i].N;
    problem_sizes_c[i].K = params.problem_sizes[i].K;
  }
  c_params.problem_sizes = problem_sizes_c.data();

  std::vector<void *> A_ptrs_c(params.A_ptrs);
  std::vector<void *> B_ptrs_c(params.B_ptrs);
  std::vector<void *> C_ptrs_c(params.C_ptrs);

  c_params.A_ptrs = A_ptrs_c.data();
  c_params.B_ptrs = B_ptrs_c.data();
  c_params.C_ptrs = C_ptrs_c.data();

  GroupedGemmResult c_result = run_grouped_gemm_2sm_c(&c_params);

  GroupedGemmResultCpp cpp_result;
  cpp_result.output_ptr = c_result.output_ptr;
  cpp_result.total_elements = c_result.total_elements;
  cpp_result.success = c_result.success;
  cpp_result.error_message = std::string(c_result.error_message);

  return cpp_result;
}

void free_output_tensor(void *ptr) { free_output_tensor_c(ptr); }

bool check_cuda_version() { return check_cuda_version_c(); }

bool check_gpu_architecture() { return check_gpu_architecture_c(); }

} // namespace grouped_gemm
#include "grouped_gemm_kernel.h"
#include <iostream>

// CUTLASS includes
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/packed_stride.hpp"

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
