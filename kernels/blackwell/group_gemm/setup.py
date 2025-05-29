import os

import torch
from pybind11 import get_cmake_dir
from pybind11.setup_helpers import build_ext, Pybind11Extension
from setuptools import Extension, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Get CUTLASS path from environment or default
cutlass_path = os.environ.get("CUTLASS_PATH", "/home/less/local/cutlas40")


# CUDA version check
cuda_version = torch.version.cuda
if cuda_version is None:
    raise RuntimeError("PyTorch was not compiled with CUDA support")

cuda_major = int(cuda_version.split(".")[0])
cuda_minor = int(cuda_version.split(".")[1])

if cuda_major < 12 or (cuda_major == 12 and cuda_minor < 8):
    raise RuntimeError("CUDA 12.8 or newer is required")

# Compiler flags
cxx_flags = [
    "-std=c++17",
    "-O3",
    "-DCUTLASS_ARCH_MMA_SM100_SUPPORTED",
    "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
]

nvcc_flags = [
    "-std=c++17",
    "-O3",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
    "-gencode=arch=compute_100,code=sm_100",  # Blackwell architecture
    "-DCUTLASS_ARCH_MMA_SM100_SUPPORTED",
    "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
]

# Include directories
include_dirs = [
    cutlass_path + "/include",
    cutlass_path + "/tools/util/include",
    cutlass_path + "/examples/common",
]

# Library directories (if needed)
library_dirs = []

# Libraries to link
libraries = ["cuda", "cudart"]

ext_modules = [
    CUDAExtension(
        "grouped_gemm_cuda",
        sources=["grouped_gemm_pytorch.cpp"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
        language="c++",
    )
]

setup(
    name="grouped_gemm_cuda",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.12.0",
        "pybind11>=2.6.0",
    ],
    zip_safe=False,
)
