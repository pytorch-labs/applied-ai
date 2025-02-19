
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
            'cxx': ['-O3'],
            'nvcc': [
                '-O3',
                '--expt-relaxed-constexpr',  # better template support
                #'-gencode=arch=compute_70,code=sm_70',  # Volta
                #'-gencode=arch=compute_75,code=sm_75',  # Turing
                #'-gencode=arch=compute_80,code=sm_80'   # Amper
                #'-gencode=arch=compute_86,code=sm_86'   # Ampere
                '-gencode=arch=compute_90,code=sm_90',  # Hopper
            ]
        })
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
