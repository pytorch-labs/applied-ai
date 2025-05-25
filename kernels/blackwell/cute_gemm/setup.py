# setup.py
import os

import pybind11
import torch
from pybind11 import get_cmake_dir
from pybind11.setup_helpers import build_ext, Pybind11Extension
from setuptools import Extension, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

# IMPORTANT: The following two lines are the only ones you need to change
# Get CUTLASS path (you'll need to set this to your CUTLASS installation)
CUTLASS_PATH = os.environ.get("CUTLASS_PATH", "/home/less/local/cutlas40")

# CUDA and PyTorch paths
cuda_home = torch.utils.cpp_extension.CUDA_HOME
pytorch_includes = torch.utils.cpp_extension.include_paths()

ext_modules = [
    CUDAExtension(
        name="sm100_gemm",
        sources=[
            "sm100_gemm_pytorch.cpp",  # PyTorch bindings (C++)
            "sm100_gemm.cu",  # CUDA kernel implementation
        ],
        include_dirs=[
            # PyTorch includes
            *pytorch_includes,
            # CUTLASS includes
            f"{CUTLASS_PATH}/include",
            f"{CUTLASS_PATH}/tools/util/include",
            # CUDA includes
            f"{cuda_home}/include",
        ],
        library_dirs=[
            f"{cuda_home}/lib64",
        ],
        libraries=["cuda", "cudart"],
        extra_compile_args={
            "cxx": [
                "-O3",
                "-std=c++17",
                "-DCUTLASS_ARCH_MMA_SM100_SUPPORTED",
                "-DCUTE_SM100_ENABLED",
            ],
            "nvcc": [
                "-O3",
                "-std=c++17",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "-gencode=arch=compute_100,code=sm_100",  # SM100 architecture
                "-DCUTLASS_ARCH_MMA_SM100_SUPPORTED",
                "-DCUTE_SM100_ENABLED",
                "--use_fast_math",
                "-Xcompiler=-fPIC",
            ],
        },
        extra_link_args=["-lcuda", "-lcudart"],
        language="c++",
    )
]

setup(
    name="sm100_gemm",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=["torch>=1.12.0"],
)
