import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

current_location = os.path.abspath(os.path.dirname(__file__))

setup(
    name='cutlass_gemm',
    ext_modules=[
        CUDAExtension(
            name='pingpong_gemm',
            sources=['cutlass.cpp', 'cutlass_kernel.cu'],
            extra_compile_args={
                'nvcc': [
                    '-DNDEBUG',
                    '-O3', 
                    '-g', 
                    '-lineinfo',
                    '--keep', 
                    '--ptxas-options=--warn-on-local-memory-usage',
                    '--ptxas-options=--warn-on-spills',
                    '--resource-usage',
                    '--source-in-ptx',
                    '-DCUTLASS_DEBUG_TRACE_LEVEL=1',
                    '-gencode=arch=compute_90a, code=sm_90a',
                ]
            },
            include_dirs=[
                f'{current_location}/cutlass/include',
                f'{current_location}/cutlass/tools/util/include',
            ],
            libraries=['cuda'],
            library_dirs=[os.path.join(CUDA_HOME, 'lib64')],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)