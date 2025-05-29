#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

// Forward declarations for opaque types (C interface)
typedef void *ElementA_C;
typedef void *ElementB_C;
typedef void *ElementC_C;

// Problem size structure (C-compatible)
typedef struct {
  int M, N, K;
} ProblemSize;

// Rasterization order enum
typedef enum { RASTER_ORDER_ALONG_M = 0, RASTER_ORDER_ALONG_N = 1 } RasterOrder;

// Parameters for grouped GEMM (C-compatible)
typedef struct {
  int num_groups;
  ProblemSize *problem_sizes;
  void **A_ptrs; // Array of ElementA_C* pointers
  void **B_ptrs; // Array of ElementB_C* pointers
  void **C_ptrs; // Array of ElementC_C* pointers
  float alpha;
  float beta;
  bool use_2sm_config;
  RasterOrder raster_order;
} GroupedGemmParams;

// Result structure (C-compatible)
typedef struct {
  void *output_ptr; // ElementC_C* pointer
  int64_t total_elements;
  bool success;
  char error_message[256];
} GroupedGemmResult;

// C function declarations (implemented in .cu file)
GroupedGemmResult run_grouped_gemm_1sm_c(const GroupedGemmParams *params);
GroupedGemmResult run_grouped_gemm_2sm_c(const GroupedGemmParams *params);
void free_output_tensor_c(void *ptr);
bool check_cuda_version_c(void);
bool check_gpu_architecture_c(void);

#ifdef __cplusplus
}
#endif

// C++ wrapper interface (only visible to C++ code)
#ifdef __cplusplus

#include <string>
#include <vector>

// CUTLASS includes for C++ only
#include "cute/tensor.hpp"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/layout/matrix.h"

namespace cutlass_grouped_gemm {

using namespace cute;

// Type definitions for C++ interface
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

// C++ wrapper structures
struct ProblemSizeCpp {
  int M, N, K;
  ProblemSizeCpp(int m, int n, int k) : M(m), N(n), K(k) {}
};

enum class RasterOrderCpp { AlongM, AlongN };

struct GroupedGemmParamsCpp {
  int num_groups;
  std::vector<ProblemSizeCpp> problem_sizes;
  std::vector<void *> A_ptrs;
  std::vector<void *> B_ptrs;
  std::vector<void *> C_ptrs;
  float alpha;
  float beta;
  bool use_2sm_config;
  RasterOrderCpp raster_order;

  GroupedGemmParamsCpp(int num_groups)
      : num_groups(num_groups), alpha(1.0f), beta(0.0f), use_2sm_config(false),
        raster_order(RasterOrderCpp::AlongM) {
    problem_sizes.reserve(num_groups);
    A_ptrs.reserve(num_groups);
    B_ptrs.reserve(num_groups);
    C_ptrs.reserve(num_groups);
  }
};

struct GroupedGemmResultCpp {
  void *output_ptr;
  int64_t total_elements;
  bool success;
  std::string error_message;

  GroupedGemmResultCpp()
      : output_ptr(nullptr), total_elements(0), success(false) {}
};

// C++ wrapper functions
GroupedGemmResultCpp run_grouped_gemm_1sm(const GroupedGemmParamsCpp &params);
GroupedGemmResultCpp run_grouped_gemm_2sm(const GroupedGemmParamsCpp &params);
void free_output_tensor(void *ptr);
bool check_cuda_version();
bool check_gpu_architecture();

} // namespace cutlass_grouped_gemm

#endif // __cplusplus
