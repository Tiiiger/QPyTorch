from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='qtorch_cuda',
    ext_modules=[
        CUDAExtension('quant_cuda', [
            'quant_cuda.cpp',
            'quant.cu',
            'fixed_point_kernel.cu',
            'block_kernel.cu',
            'float_kernel.cu',
            'sim_helper.cu',
            'bit_helper.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
