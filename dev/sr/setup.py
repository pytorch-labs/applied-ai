from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='stochastic_rounding_cuda',
    ext_modules=[
        CUDAExtension('stochastic_rounding_cuda', [
            'src/stochastic_rounding.cpp',
            'src/stochastic_rounding.cu'
        ],
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': ['-O3']
        })
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
