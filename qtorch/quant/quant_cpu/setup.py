from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(name='qtorch_cpu',
      ext_modules=[CppExtension('quant_cpu', ['quant_cpu.cpp'])],
      cmdclass={'build_ext': BuildExtension})
