# QPyTorch

In this fork from [QPyTorch](https://github.com/Tiiiger/QPyTorch), I am experimenting with updating the quantization operations implemented for CPU such that they are registered for TrochScript according to the official PyTorch documentation for [extending TorchScript with custom C++ operators](https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html#registering-the-custom-operator-with-torchscript) instead of using `PYBIND11`.

This fork also includes an experiment on creating a basic Torch module and compiling it down using Torch-MLIR (details can be found [here](https://github.com/llvm/torch-mlir/issues/910)). 

## Installation

In addition to the basic [requirements for QPyTorch](https://github.com/Tiiiger/QPyTorch#installation), packages below are needed to run the experiment:

- wheel
- numpy
- An installation of Torch-MLIR

## Torch-MLIR Examperiment

First make sure you have the changes similar to this PR, and then
1. Build the custom operators for `qtorch_ops` by creating some directory named `build` and running the CMake command `cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" $PATH_TO_QUANT-CPU_DIR` in it. Where `PATH_TO_QUANT-CPU_DIR` is the path to `QPyTorch/qtroch/quant/quant_cpu`.
1. Run `make` to create the shared library `libqtorch_ops.so`.
1. Build `qtorch` package using `pip install .` command from the top-level directory in QPyTorch.
1. Setup Python environment for Torch-MLIR using the command below:
```
export PYTHONPATH=`pwd`/build/tools/torch-mlir/python_packages/torch_mlir:`pwd`/examples
```
1. Run the Torch-MLIR experiment script in `examples/torch-mlir_experiment.py`