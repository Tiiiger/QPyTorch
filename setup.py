from setuptools import setup, find_packages

try:
    import torch
    has_dev_pytorch = "dev" in torch.__version__
except ImportError:
    has_dev_pytorch = False

# Base equirements
install_requires = [
    "torch>=1.0.0",
]
if has_dev_pytorch:  # Remove the PyTorch requirement
    install_requires = [
        install_require for install_require in install_requires
        if "torch" != re.split(r"(=|<|>)", install_require)[0]
    ]

setup(name='qtorch',
      version='0.1.0',
      description="Low-Precision Arithmetic Simulation in Pytorch",
      long_description=open("README.md").read(),
      author="Tianyi Zhang, Zhiqiu Lin, Guandao Yang, Christopher De Sa",
      author_email="tz58@cornell.edu",
      project_urls={
        "Documentation": "https://qpytorch.readthedocs.io",
        "Source": "https://github.com/Tiiiger/QPyTorch/graphs/contributors",
      },
      packages=find_packages(),
      license="MIT",
      python_requires=">=3.6",
      install_requires=install_requires
)
