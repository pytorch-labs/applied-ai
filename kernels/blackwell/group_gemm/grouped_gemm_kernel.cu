#include "grouped_gemm_kernel.h"
#include <cstring>
#include <cuda_runtime.h>

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
using ClusterShape = Shape<_1, _1, _1>; // Start with 1x1x1 cluster

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
GroupedGemmResult run_gemm_impl(const GroupedGemmParams *params) {
  GroupedGemmResult result = {};
  result.success = false;

  try {
    const int num_groups = params->num_groups;

    // Prepare problem sizes
    std::vector<typename ProblemShape::UnderlyingProblemShape>
        problem_sizes_host;
    int64_t total_elements = 0;

    for (int i = 0; i < num_groups; ++i) {
      problem_sizes_host.push_back({params->problem_sizes[i].M,
                                    params->problem_sizes[i].N,
                                    params->problem_sizes[i].K});
      total_elements += params->problem_sizes[i].M * params->problem_sizes[i].N;
    }

    // Allocate output
    ElementC *output_ptr = nullptr;
    cudaError_t cuda_error =
        cudaMalloc(&output_ptr, total_elements * sizeof(ElementC));
    if (cuda_error != cudaSuccess) {
      snprintf(result.error_message, sizeof(result.error_message),
               "Failed to allocate output: %s", cudaGetErrorString(cuda_error));
      return result;
    }

    // Prepare output pointers
    std::vector<ElementC *> D_ptrs_host;
    int64_t offset = 0;
    for (int i = 0; i < num_groups; ++i) {
      D_ptrs_host.push_back(output_ptr + offset);
      offset += params->problem_sizes[i].M * params->problem_sizes[i].N;
    }

    // Prepare strides - use proper types for each Gemm variant
    using StrideA = typename Gemm::GemmKernel::InternalStrideA;
    using StrideB = typename Gemm::GemmKernel::InternalStrideB;
    using StrideC = typename Gemm::GemmKernel::InternalStrideC;
    using StrideD = typename Gemm::GemmKernel::InternalStrideD;

    std::vector<StrideA> stride_A_host;
    std::vector<StrideB> stride_B_host;
    std::vector<StrideC> stride_C_host;
    std::vector<StrideD> stride_D_host;

    for (int i = 0; i < num_groups; ++i) {
      int M = params->problem_sizes[i].M;
      int N = params->problem_sizes[i].N;
      int K = params->problem_sizes[i].K;

      stride_A_host.push_back(
          cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1}));
      stride_B_host.push_back(
          cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1}));
      stride_C_host.push_back(
          cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1}));
      stride_D_host.push_back(
          cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1}));
    }

    // Device allocations
    cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape>
        problem_sizes_device(num_groups);
    problem_sizes_device.copy_from_host(problem_sizes_host.data());

    cutlass::DeviceAllocation<const ElementA *> ptr_A_device(num_groups);
    ptr_A_device.copy_from_host(
        reinterpret_cast<const ElementA **>(params->A_ptrs));

    cutlass::DeviceAllocation<const ElementB *> ptr_B_device(num_groups);
    ptr_B_device.copy_from_host(
        reinterpret_cast<const ElementB **>(params->B_ptrs));

    cutlass::DeviceAllocation<const ElementC *> ptr_C_device(num_groups);
    ptr_C_device.copy_from_host(
        reinterpret_cast<const ElementC **>(params->C_ptrs));

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
    cudaGetDevice(&hw_info.device_id);
    hw_info.sm_count =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
            hw_info.device_id);

    // Set cluster shape based on configuration
    if constexpr (!cute::is_static_v<ClusterShape>) {
      hw_info.cluster_shape = dim3(params->use_2sm_config ? 2 : 1, 1, 1);
      hw_info.cluster_shape_fallback = dim3(1, 1, 1);
    }

    // Epilogue arguments
    typename Gemm::Arguments::EpilogueArguments epilogue_args{};
    epilogue_args.thread = {params->alpha, params->beta};
    epilogue_args.ptr_C = ptr_C_device.get();
    epilogue_args.dC = stride_C_device.get();
    epilogue_args.ptr_D = ptr_D_device.get();
    epilogue_args.dD = stride_D_device.get();

    // Scheduler arguments
    typename Gemm::GemmKernel::TileSchedulerArguments scheduler_args{};
    scheduler_args.raster_order =
        (params->raster_order == RASTER_ORDER_ALONG_N)
            ? cutlass::gemm::kernel::detail::
                  PersistentTileSchedulerSm100GroupParams<
                      typename ProblemShape::UnderlyingProblemShape>::
                      RasterOrderOptions::AlongN
            : cutlass::gemm::kernel::detail::
                  PersistentTileSchedulerSm100GroupParams<
                      typename ProblemShape::UnderlyingProblemShape>::
                      RasterOrderOptions::AlongM;

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
      snprintf(result.error_message, sizeof(result.error_message),
               "CUTLASS cannot implement arguments: %s",
               cutlass::cutlassGetStatusString(status));
      return result;
    }

    status = gemm.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
      cudaFree(output_ptr);
      snprintf(result.error_message, sizeof(result.error_message),
               "CUTLASS initialization failed: %s",
               cutlass::cutlassGetStatusString(status));
      return result;
    }

    status = gemm.run();
    if (status != cutlass::Status::kSuccess) {
      cudaFree(output_ptr);
      snprintf(result.error_message, sizeof(result.error_message),
               "CUTLASS execution failed: %s",
               cutlass::cutlassGetStatusString(status));
      return result;
    }

    cuda_error = cudaDeviceSynchronize();
    if (cuda_error != cudaSuccess) {
      cudaFree(output_ptr);
      snprintf(result.error_message, sizeof(result.error_message),
               "CUDA synchronization failed: %s",
               cudaGetErrorString(cuda_error));
      return result;
    }

    result.output_ptr = output_ptr;
    result.total_elements = total_elements;
    result.success = true;
    strcpy(result.error_message, "Success");

  } catch (const std::exception &e) {
    snprintf(result.error_message, sizeof(result.error_message),
             "Exception: %s", e.what());
  } catch (...) {
    strcpy(result.error_message, "Unknown error");
  }

  return result;
}

#endif // CUTLASS_ARCH_MMA_SM100_SUPPORTED

// C interface implementations
extern "C" {

GroupedGemmResult run_grouped_gemm_1sm_c(const GroupedGemmParams *params) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  return run_gemm_impl<Gemm1SM>(params);
#else
  GroupedGemmResult result = {};
  strcpy(result.error_message, "CUTLASS SM100 support not available");
  return result;
#endif
}

GroupedGemmResult run_grouped_gemm_2sm_c(const GroupedGemmParams *params) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  return run_gemm_impl<Gemm2SM>(params);
#else
  GroupedGemmResult result = {};
  strcpy(result.error_message, "CUTLASS SM100 support not available");
  return result;
#endif
}

extern "C" void free_output_tensor_c(void *ptr) {
  if (ptr) {
    cudaFree(ptr);
  }
}

bool check_cuda_version_c(void) {
#if defined(__CUDACC_VER_MAJOR__) && defined(__CUDACC_VER_MINOR__)
  return (__CUDACC_VER_MAJOR__ > 12) ||
         (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 8);
#else
  return false;
#endif
}

bool check_gpu_architecture_c(void) {
  cudaDeviceProp props;
  int current_device_id;

  if (cudaGetDevice(&current_device_id) != cudaSuccess)
    return false;

  if (cudaGetDeviceProperties(&props, current_device_id) != cudaSuccess)
    return false;

  return (props.major == 10 && props.minor == 0);
}

} // extern "C"
