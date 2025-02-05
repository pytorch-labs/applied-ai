cmake_minimum_required(VERSION 3.18)
project(grouped_gemm CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 90)

# Find CUTLASS
if(NOT DEFINED ENV{CUTLASS_PATH})
    message(FATAL_ERROR "CUTLASS_PATH environment variable is not set")
endif()
set(CUTLASS_PATH $ENV{CUTLASS_PATH})
set(CUTLASS_INCLUDE_DIRS
    ${CUTLASS_PATH}/include
    ${CUTLASS_PATH}/tools/util/include
)

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Add test executable
add_executable(test_grouped_gemm
    test_grouped_gemm.cu
)

target_include_directories(test_grouped_gemm PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUTLASS_INCLUDE_DIRS}
)

# Set CUDA compilation flags
target_compile_options(test_grouped_gemm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --expt-relaxed-constexpr
        -std=c++17
        --extended-lambda
        --use_fast_math
    >
)

# Link against CUDA libraries
target_link_libraries(test_grouped_gemm PRIVATE
    cudart
    cublas
)
