from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='quantizer',
    ext_modules=[
        CUDAExtension('quant_cuda', [
            'quant_cuda.cpp',
            'quant_kernel.cu',
            'extract_exponent.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
