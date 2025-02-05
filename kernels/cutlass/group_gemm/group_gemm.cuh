#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/epilogue/thread/linear_combination.h"

namespace cggg {
namespace group_gemm {

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementOutput = cutlass::half_t;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::ColumnMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::ColumnMajor;

// Define the GEMM kernel using CUTLASS's recommended configuration
using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    ElementA,
    LayoutA,
    cutlass::ComplexTransform::kNone,
    8,
    ElementB,
    LayoutB,
    cutlass::ComplexTransform::kNone,
    8,
    ElementOutput,
    LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator,
        ElementAccumulator>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    4
>::GemmKernel;

using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;

cudaError_t CutlassGroupedGEMM(
    const cutlass::gemm::GemmCoord* problem_sizes,
    int problem_count,
    half** ptr_A,
    half** ptr_B,
    half** ptr_C,
    int64_t* lda,
    int64_t* ldb,
    int64_t* ldc,
    cudaStream_t stream = nullptr) {

    // Create epilogue operation
    typename GemmGrouped::EpilogueOutputOp::Params epilogue_op(1.0f, 0.0f);

    // Get recommended threadblock count
    int threadblock_count = GemmGrouped::sufficient(
        reinterpret_cast<const cutlass::gemm::GemmCoord*>(problem_sizes),
        problem_count);

    if (!threadblock_count) {
        return cudaErrorInvalidValue;
    }

    // Initialize GEMM arguments
    typename GemmGrouped::Arguments args(
        reinterpret_cast<cutlass::gemm::GemmCoord*>(const_cast<cutlass::gemm::GemmCoord*>(problem_sizes)),
        problem_count,
        threadblock_count,
        epilogue_op,
        reinterpret_cast<ElementA**>(ptr_A),
        reinterpret_cast<ElementB**>(ptr_B),
        reinterpret_cast<ElementOutput**>(ptr_C),
        reinterpret_cast<ElementOutput**>(ptr_C),
        lda,
        ldb,
        ldc,
        ldc
    );

    // Initialize and run GEMM
    GemmGrouped gemm_op;
    
    cutlass::Status status = gemm_op.initialize(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorInvalidValue;
    }

    status = gemm_op.run(stream);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}

} // namespace group_gemm
} // namespace cggg
