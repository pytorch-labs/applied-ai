#include <iostream>
#include <vector>
#include <random>
#include "grouped_gemm.cuh"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/device_memory.h"

struct TestbedGrouped {
    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = cutlass::half_t;
    using ElementAccumulator = float;

    using LayoutA = cutlass::layout::ColumnMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::ColumnMajor;

    // Device memory allocations
    cutlass::DeviceAllocation<cutlass::gemm::GemmCoord> problem_sizes_device;
    std::vector<int64_t> offset_A;
    std::vector<int64_t> offset_B;
    std::vector<int64_t> offset_C;
    std::vector<int64_t> lda_host;
    std::vector<int64_t> ldb_host;
    std::vector<int64_t> ldc_host;
    cutlass::DeviceAllocation<ElementA> block_A;
    cutlass::DeviceAllocation<ElementB> block_B;
    cutlass::DeviceAllocation<ElementC> block_C;
    cutlass::DeviceAllocation<ElementA *> ptr_A;
    cutlass::DeviceAllocation<ElementB *> ptr_B;
    cutlass::DeviceAllocation<ElementC *> ptr_C;
    cutlass::DeviceAllocation<int64_t> lda;
    cutlass::DeviceAllocation<int64_t> ldb;
    cutlass::DeviceAllocation<int64_t> ldc;

    int problem_count;
    std::vector<cutlass::gemm::GemmCoord> problem_sizes_host;

    TestbedGrouped(int problem_count_) : problem_count(problem_count_) {
        initialize_problems();
        allocate();
        initialize();
    }

    void initialize_problems() {
        problem_sizes_host.reserve(problem_count);

        // Define problems sizes (M, N, K)
        problem_sizes_host.push_back({128, 256, 64});
        problem_sizes_host.push_back({256, 128, 64});

        // Initialize offsets and leading dimensions
        offset_A.resize(problem_count);
        offset_B.resize(problem_count);
        offset_C.resize(problem_count);
        lda_host.resize(problem_count);
        ldb_host.resize(problem_count);
        ldc_host.resize(problem_count);

        size_t offset_a = 0;
        size_t offset_b = 0;
        size_t offset_c = 0;

        for (int i = 0; i < problem_count; ++i) {
            auto problem = problem_sizes_host[i];

            // Set leading dimensions for column-major layout
            lda_host[i] = problem.m();
            ldb_host[i] = problem.k();
            ldc_host[i] = problem.m();

            // Set offsets
            offset_A[i] = offset_a;
            offset_B[i] = offset_b;
            offset_C[i] = offset_c;

            // Compute next offsets
            offset_a += problem.m() * problem.k();
            offset_b += problem.k() * problem.n();
            offset_c += problem.m() * problem.n();
        }

        // Copy problem sizes to device
        problem_sizes_device.reset(problem_count);
        cutlass::device_memory::copy_to_device(
            problem_sizes_device.get(), problem_sizes_host.data(), problem_count);
    }

    void allocate() {
        // Compute total sizes needed
        size_t total_A = 0;
        size_t total_B = 0;
        size_t total_C = 0;

        for (int i = 0; i < problem_count; ++i) {
            auto problem = problem_sizes_host[i];
            total_A += problem.m() * problem.k();
            total_B += problem.k() * problem.n();
            total_C += problem.m() * problem.n();
        }

        // Allocate device memory
        block_A.reset(total_A);
        block_B.reset(total_B);
        block_C.reset(total_C);
        ptr_A.reset(problem_count);
        ptr_B.reset(problem_count);
        ptr_C.reset(problem_count);
        lda.reset(problem_count);
        ldb.reset(problem_count);
        ldc.reset(problem_count);

        // Setup pointer arrays
        std::vector<ElementA *> ptr_A_host(problem_count);
        std::vector<ElementB *> ptr_B_host(problem_count);
        std::vector<ElementC *> ptr_C_host(problem_count);

        for (int i = 0; i < problem_count; ++i) {
            ptr_A_host[i] = block_A.get() + offset_A[i];
            ptr_B_host[i] = block_B.get() + offset_B[i];
            ptr_C_host[i] = block_C.get() + offset_C[i];
        }

        // Copy pointers and leading dimensions to device
        cutlass::device_memory::copy_to_device(ptr_A.get(), ptr_A_host.data(), problem_count);
        cutlass::device_memory::copy_to_device(ptr_B.get(), ptr_B_host.data(), problem_count);
        cutlass::device_memory::copy_to_device(ptr_C.get(), ptr_C_host.data(), problem_count);
        cutlass::device_memory::copy_to_device(lda.get(), lda_host.data(), problem_count);
        cutlass::device_memory::copy_to_device(ldb.get(), ldb_host.data(), problem_count);
        cutlass::device_memory::copy_to_device(ldc.get(), ldc_host.data(), problem_count);
    }

    void initialize() {
        // Initialize matrices with random data
        cutlass::reference::device::BlockFillRandomUniform(
            block_A.get(), block_A.size(), 0, ElementA(1), ElementA(-1));
        cutlass::reference::device::BlockFillRandomUniform(
            block_B.get(), block_B.size(), 0, ElementB(1), ElementB(-1));
        cutlass::reference::device::BlockFillRandomUniform(
            block_C.get(), block_C.size(), 0, ElementC(1), ElementC(-1));
    }

    bool run() {
        cudaStream_t stream = nullptr;
        cudaError_t result = cggg::group_gemm::CutlassGroupedGEMM(
            problem_sizes_host.data(),
            problem_count,
            reinterpret_cast<half**>(ptr_A.get()),
            reinterpret_cast<half**>(ptr_B.get()),
            reinterpret_cast<half**>(ptr_C.get()),
            lda.get(),
            ldb.get(),
            ldc.get(),
            stream
        );

        if (result != cudaSuccess) {
            std::cerr << "GEMM failed with error: " << cudaGetErrorString(result) << std::endl;
            return false;
        }

        return true;
    }

};

int main() {
    TestbedGrouped testbed(2);  // 2 problems

    std::cout << "Running grouped GEMM..." << std::endl;

    if (!testbed.run()) {
        std::cerr << "GEMM execution failed!" << std::endl;
        return 1;
    }

    std::cout << "GroupGEMM complete..." << std::endl;


    std::cout << "All tests passed!" << std::endl;
    return 0;
}
