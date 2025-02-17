from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='stochastic_round_cuda',
    ext_modules=[
        CUDAExtension('stochastic_round_cuda', [
            'src/sr_cuda.cpp',
            'src/sr_kernel.cu',
        ],
        extra_compile_args={
            'cxx': [],
            'nvcc': [
                '--use_fast_math',
                '--expt-relaxed-constexpr',
                '-O3',
            ]
        })
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
