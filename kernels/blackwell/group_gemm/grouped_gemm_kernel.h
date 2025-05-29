#pragma once

#include <cuda_runtime.h>
#include <vector>

// CUTLASS type definitions

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

// Type definitions
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using ElementC = cutlass::half_t;
using ElementAccumulator = float;

// Layout configurations
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::ColumnMajor;

constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

// Architecture and operator configuration
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using ClusterShape = Shape<int32_t, int32_t, _1>;

// Forward declare stride types
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
using StrideA = cutlass::detail::TagToStrideA_t<LayoutA>;
using StrideB = cutlass::detail::TagToStrideB_t<LayoutB>;
using StrideC = cutlass::detail::TagToStrideC_t<LayoutC>;
using StrideD = cutlass::detail::TagToStrideC_t<LayoutC>;
#else
// Placeholder types when CUTLASS SM100 is not supported
struct DummyStride {};
using StrideA = DummyStride;
using StrideB = DummyStride;
using StrideC = DummyStride;
using StrideD = DummyStride;
#endif

// CUDA error checking macro
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      throw std::runtime_error("CUDA error: " +                                \
                               std::string(cudaGetErrorString(error)));        \
    }                                                                          \
  } while (0)

namespace grouped_gemm {

// Problem size structure
struct ProblemSize {
  int M, N, K;

  ProblemSize(int m, int n, int k) : M(m), N(n), K(k) {}
};

// Rasterization order enum
enum class RasterOrder { AlongM, AlongN };

// Parameters for grouped GEMM
struct GroupedGemmParams {
  int num_groups;
  std::vector<ProblemSize> problem_sizes;
  std::vector<ElementA *> A_ptrs;
  std::vector<ElementB *> B_ptrs;
  std::vector<ElementC *> C_ptrs;
  float alpha;
  float beta;
  bool use_2sm_config;
  RasterOrder raster_order;

  GroupedGemmParams(int num_groups)
      : num_groups(num_groups), alpha(1.0f), beta(0.0f), use_2sm_config(false),
        raster_order(RasterOrder::AlongM) {
    problem_sizes.reserve(num_groups);
    A_ptrs.reserve(num_groups);
    B_ptrs.reserve(num_groups);
    C_ptrs.reserve(num_groups);
  }
};

// Result structure
struct GroupedGemmResult {
  ElementC *output_ptr;
  int64_t total_elements;
  bool success;

  GroupedGemmResult()
      : output_ptr(nullptr), total_elements(0), success(false) {}
};

// Function declarations
GroupedGemmResult run_grouped_gemm_1sm(const GroupedGemmParams &params);
GroupedGemmResult run_grouped_gemm_2sm(const GroupedGemmParams &params);
void free_output_tensor(ElementC *ptr);

} // namespace grouped_gemm
