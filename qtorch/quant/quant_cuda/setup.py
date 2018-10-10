from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='qtorch_cuda',
    ext_modules=[
        CUDAExtension('quant_cuda', [
            'quant_cuda.cpp',
            'quant_kernel.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
