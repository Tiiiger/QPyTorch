# Stochastic Weight Averaging for Low-Precision Training (SWALP)

This repository contains a QPyTorch implementation of the paper:

[SWALP : Stochastic Weight Averaging for Low-Precision Training (SWALP)](https://arxiv.org/abs/1904.11943). 

[Guandao Yang](http://www.guandaoyang.com), 
[Tianyi Zhang](https://tiiiger.github.io/), 
Polina Kirichenko, 
[Junwen Bai](http://www.cs.cornell.edu/~junwen/), 
[Andrew Gordon Wilson](https://people.orie.cornell.edu/andrew/), 
[Christopher De Sa](http://www.cs.cornell.edu/~cdesa/)

![swalp-image](https://github.com/stevenygd/SWALP/blob/master/assets/swalp.jpg)

## Introduction

Low precision operations can provide scalability, memory savings, portability,
and energy efficiency. SWALP averages low-precision SGD iterates with a modified
learning rate schedule and can match the performance of *full-precision* SGD
even with all numbers quantized down to 8 bits, including the gradient
accumulators.

This repo implements SWALP in QPyTorch and provides and example on VGG. 
In training a simulated low-precision VGG model, each epoch takes 23.0s in
this implementation whereas the PyTorch implementation takes 36.8s (measured on
a GTX 1080-Ti GPU).

## Citation
Please cite the SWALP paper if you find this repo useful in your research:
```latex
@misc{gu2019swalp,
    title={SWALP : Stochastic Weight Averaging in Low-Precision Training},
    author={Guandao Yang and Tianyi Zhang and Polina Kirichenko and Junwen Bai and Andrew Gordon Wilson and Christopher De Sa},
    year={2019},
    eprint={1904.11943},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

## Dependencies
* [PyTorch](http://pytorch.org/) >= 1.1
* [torchvision](https://github.com/pytorch/vision/)
* [tensorflow](https://www.tensorflow.org/) to use tensorboard
* [qtorch](https://github.com/Tiiiger/QPyTorch) >= 0.1.1

To install other requirements through `$ pip install -r requirements.txt`.

## Usage
```bash
bash example.sh
```

## Results
| Datset   | Model        | SGD-FP     | SWA-FP     | SGD-LP     | SWALP      |
|----------|--------------|------------|------------|------------|------------|
| CIFAR10  | VGG16        | 6.81±0.09  | 6.51±0.14  | 7.61±0.15  | 6.70±0.12  |
| CIFAR100 | VGG16        | 27.23±0.17 | 25.93±0.21 | 29.59±0.32 | 26.65±0.29 |

## References
This repo is modified from the PyTorch repo of [SWALP](https://github.com/stevenygd/SWALP)
Network architecture implementations are adapted from:
VGG: [github.com/pytorch/vision/](https://github.com/pytorch/vision/)
