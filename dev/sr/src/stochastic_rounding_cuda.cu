#include "stochastic_rounding.hpp"
#include <cstdint>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// global defines
#define THREADS_PER_BLOCK 256
#define ELEMENTS_PER_THREAD 8
#define VECTOR_SIZE 4
#define WARP_SIZE 32

// core atomic helper function for correct rounding
__device__ __forceinline__ __nv_bfloat16
float_to_bf16_stochastic(float value, uint32_t rand32) {
    uint32_t value_bits = __float_as_uint(value);
    uint32_t truncated = value_bits & 0xFFFF;
    uint32_t rounded = value_bits & 0xFFFF0000u;

    bool should_round_up = (rand32 & 0xFFFF) < truncated;
    if (should_round_up) {
        rounded += 0x10000;
    }

    return __float2bfloat16(__uint_as_float(rounded));
}

// Vectorized conversion with proper rounding
__device__ __forceinline__ void float4_to_bf16_4_stochastic(
    const float4& input,
    __nv_bfloat16* output,
    const float4& rand4) {

    const float values[4] = {input.x, input.y, input.z, input.w};

    // Convert random floats to uint32 once
    const uint32_t rand_bits[4] = {
        __float_as_uint(rand4.x),
        __float_as_uint(rand4.y),
        __float_as_uint(rand4.z),
        __float_as_uint(rand4.w)
    };

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint32_t value_bits = __float_as_uint(values[i]);
        uint32_t truncated = value_bits & 0xFFFF;
        uint32_t rounded = value_bits & 0xFFFF0000u;

        bool should_round_up = (rand_bits[i] & 0xFFFF) < truncated;
        if (should_round_up) {
            rounded += 0x10000;
        }

        output[i] = __float2bfloat16(__uint_as_float(rounded));
    }
}

// Prefetch helper
__device__ __forceinline__ void prefetch_float4(const float4* addr) {
    asm("prefetch.global.L2 [%0];": : "l"(addr));
}

__global__ void __launch_bounds__(THREADS_PER_BLOCK, 4)
stochastic_round_bf16_kernel(const float4* __restrict__ input,
                      __nv_bfloat16* __restrict__ output,
                      const int size,
                      const unsigned long long seed) {

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WARP_SIZE>(block);

    // Initialize random state
    curandStatePhilox4_32_10_t state;
    const unsigned long long unique_id = (blockIdx.x * blockDim.x + threadIdx.x);
    curand_init(seed, unique_id, 0, &state);

    const int tid = block.thread_rank();
    const int idx = blockIdx.x * blockDim.x + tid;
    const int stride = blockDim.x * gridDim.x;
    const int size4 = size / 4;

    // Main processing loop
    for (int i = idx; i < size4; i += stride * ELEMENTS_PER_THREAD) {
        // Process ELEMENTS_PER_THREAD float4s per thread
        #pragma unroll
        for (int j = 0; j < ELEMENTS_PER_THREAD; j++) {
            const int current_idx = i + j * stride;
            if (current_idx < size4) {
                // Prefetch next chunk
                if (j < ELEMENTS_PER_THREAD - 1) {
                    prefetch_float4(&input[current_idx + stride]);
                }

                // Load input data
                float4 input4 = input[current_idx];
                float4 rand4 = curand_uniform4(&state);

                // Process and store with correct rounding
                float4_to_bf16_4_stochastic(
                    input4,
                    &output[current_idx * 4],
                    rand4
                );
            }
        }
    }

    // Handle remaining elements with single rounding function
    const int remaining = size % (4 * VECTOR_SIZE);
    if (remaining > 0) {
        // Only one warp handles remaining elements
        if (warp.meta_group_rank() == 0) {
            const int base = size - remaining;
            const float* input_float = reinterpret_cast<const float*>(&input[base / 4]);

            // Process remaining elements
            for (int i = warp.thread_rank(); i < remaining; i += WARP_SIZE) {
                float rand_val = curand_uniform(&state);
                output[base + i] = float_to_bf16_stochastic(
                    input_float[i],
                    __float_as_uint(rand_val * UINT32_MAX)
                );
            }
        }
    }
}

torch::Tensor stochastic_round_bf16_cuda(torch::Tensor input, bool requires_grad = false) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input tensor must be float32");

    const int num_elements = input.numel();
    const int num_elements4 = (num_elements + 3) / 4;

    // Calculate grid size
    const int vector_elements = VECTOR_SIZE * ELEMENTS_PER_THREAD;
    const int elements_per_block = THREADS_PER_BLOCK * vector_elements;
    const int num_blocks = (num_elements + elements_per_block - 1) / elements_per_block;
    const int max_blocks = 128000;
    const int blocks = std::min(num_blocks, max_blocks);

    // Create output tensor
    auto options = torch::TensorOptions()
                      .dtype(torch::kBFloat16)
                      .device(input.device())
                      .requires_grad(requires_grad)
                      .memory_format(torch::MemoryFormat::Contiguous);
    auto output = torch::empty_like(input, options);

    // Generate random seed
    std::random_device rd;
    std::seed_seq seq{rd(), rd(), rd(), rd()};
    std::mt19937_64 gen(seq);
    std::uniform_int_distribution<unsigned long long> dis;
    const unsigned long long seed = dis(gen);

    // Set cache preference
    cudaFuncSetCacheConfig(stochastic_round_bf16_kernel, cudaFuncCachePreferL1);

    // Launch kernel
    stochastic_round_bf16_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        reinterpret_cast<const float4*>(input.data_ptr<float>()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        num_elements,
        seed
    );

    // Error checking
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel execution failed: ", cudaGetErrorString(err));

    err = cudaDeviceSynchronize();
    TORCH_CHECK(err == cudaSuccess, "CUDA synchronization failed: ", cudaGetErrorString(err));

    return output;
}
