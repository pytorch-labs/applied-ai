
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='stochastic_rounding_cuda',
    version='0.1.021825',
    ext_modules=[
        CUDAExtension('stochastic_rounding_cuda', [
            'src/stochastic_rounding.cu',
            'src/stochastic_rounding_cuda.cu'
        ],
        extra_compile_args={
            'cxx': ['-O3', '-march=native', '-ffast-math'],
            'nvcc': [
                '-O3',
                '--use_fast_math',
                '--ptxas-options=-v',
                '--maxrregcount=32',
                '--restrict',
                '--extra-device-vectorization',
                '--expt-relaxed-constexpr',
                '-gencode=arch=compute_90,code=sm_90',
            ]
        })
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
