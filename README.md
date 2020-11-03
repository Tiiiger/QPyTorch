# Extended version of QPytorch
#### Author: minhhn2910@github, himeshi@github
#### Install in developer mode: 
```bash
git clone https://github.com/minhhn2910/QPyTorch.git
cd QPyTorch
pip install -e ./
```
simple test if c-extension is working correctly : 
```
python test.py
```
Important: if there are errors when running test.py, please export the environment variables indicating build directory and/or CUDA_HOME, otherwise we will have permission problem in multi-user-server.
```
export TORCH_EXTENSIONS_DIR=/[your-home-folder]/torch_extension
export CUDA_HOME=/[your cuda instalation directory e.g. /usr/local/cuda-10.2] 
python test.py
```

#### Functionality: 
#### Currently under development
#### Live demo of applying posit 6 bits & 8 bits to [ALBERT](https://huggingface.co/ktrapeznikov/albert-xlarge-v2-squad-v2) for Question Answering Task: [GoogleColab Demo](https://colab.research.google.com/drive/1t2bsoQb4oI-Lind_ORzroyv8X2H78cdn?usp=sharing)  
#### Most functionalities can be tested by using the notebooks in posit tutorials: ./examples/tutorial/
#### The below is the original README file 

# QPyTorch
[![Downloads](https://pepy.tech/badge/qtorch)](https://pepy.tech/project/qtorch) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

#### News:
- Updated to version 0.2.0:
  - **Bug fixed**: previously in our floating point quantization, numbers that are closer to 0 than the smallest 
  representable positive number are rounded to the smallest rep positive number. Now we round to 0 or the smallest 
  representable number based on which one is the nearest.
  - **Different Behavior**: To be consistent with PyTorch [Issue #17443](https://github.com/pytorch/pytorch/pull/17443),
  we round to nearest even now.
  - We migrate to PyTorch 1.5.0. There are several changes in the C++ API of PyTorch. 
  This new version is not backward-compatible with older PyTorch. 
  - *Note*: if you are using CUDA 10.1, please install CUDA 10.1 Update 1 (or later version). There is a bug in 
  the first version of CUDA 10.1 which leads to compilation errors.
  - *Note*: previous users, please remove the cache in the pytorch extension directory. 
  For example, you can run this command `rm -rf /tmp/torch_extensions/quant_cuda /tmp/torch_extensions/quant_cpu` if 
  you are using the default directory for pytorch extensions.

# Overview
QPyTorch is a low-precision arithmetic simulation package in
PyTorch. It is designed to support researches on low-precision machine
learning, especially for researches in low-precision training. 
A more comprehensive write-up can be found [here](https://arxiv.org/abs/1910.04540).

Notably, QPyTorch supports quantizing different numbers in the training process
with customized low-precision formats. This eases the process of investigating
different precision settings and developing new deep learning architectures. More
concretely, QPyTorch implements fused kernels for quantization and integrates
smoothly with existing PyTorch kernels (e.g. matrix multiplication, convolution). 

Recent researches can be reimplemented easily through QPyTorch. We offer an
example replication of [WAGE](https://arxiv.org/abs/1802.04680) in a downstream
repo [WAGE](https://github.com/Tiiiger/QPyTorch/blob/master/examples/WAGE). We also provide a list
of working examples under [Examples](#examples).

*Note*: QPyTorch relies on PyTorch functions for the underlying computation,
such as matrix multiplication. This means that the actual computation is done in
single precision. Therefore, QPyTorch is not intended to be used to study the
numerical behavior of different **accumulation** strategies.

*Note*: QPyTorch, as of now, have a different rounding mode with PyTorch. QPyTorch does round-away-from-zero while
PyTorch does round-to-nearest-even. This will create a discrepancy between the PyTorch half-precision tensor 
and QPyTorch's simulation of half-precision numbers.

if you find this repo useful please cite
```
@misc{zhang2019qpytorch,
    title={QPyTorch: A Low-Precision Arithmetic Simulation Framework},
    author={Tianyi Zhang and Zhiqiu Lin and Guandao Yang and Christopher De Sa},
    year={2019},
    eprint={1910.04540},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

## Installation

requirements:

- Python >= 3.6
- PyTorch >= 1.5.0
- GCC >= 4.9 on linux
- CUDA >= 10.1 on linux

Install other requirements by:
```bash
pip install -r requirements.txt
```

Install QPyTorch through pip:
```bash
pip install qtorch
```

For more details about compiler requirements, 
please refer to [PyTorch extension tutorial](https://pytorch.org/tutorials/advanced/cpp_extension.html).

## Documentation
See our [readthedocs](https://qpytorch.readthedocs.io/en/latest/) page.

## Tutorials
- [An overview of QPyTorch's features](https://github.com/Tiiiger/QPyTorch/blob/master/examples/tutorial/Functionality_Overview.ipynb)
- [CIFAR-10 Low-Precision Training Tutorial](https://github.com/Tiiiger/QPyTorch/blob/master/examples/tutorial/CIFAR10_Low_Precision_Training_Example.ipynb)

## Examples
- Low-Precision VGGs and ResNets using fixed point, block floating point on CIFAR and ImageNet. [lp_train](https://github.com/Tiiiger/QPyTorch/blob/master/examples/lp_train)
- Reproduction of WAGE in QPyTorch. [WAGE](https://github.com/Tiiiger/QPyTorch/blob/master/examples/WAGE)
- Implementation (simulation) of 8-bit Floating Point Training in QPyTorch. [IBM8](https://github.com/Tiiiger/QPyTorch/blob/master/examples/IBM8)

## Team
* [Tianyi Zhang](https://scholar.google.com/citations?user=OI0HSa0AAAAJ&hl=en)
* Zhiqiu Lin
* [Guandao Yang](http://www.guandaoyang.com/)
* [Christopher De Sa](http://www.cs.cornell.edu/~cdesa/)
