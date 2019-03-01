# Stochastic Weight Averaging for Low-Precision Training (SWALP)

This repository contains a PyTorch implementation of the paper:

[SWALP : Stochastic Weight Averaging for Low-Precision Training (SWALP)](www.arxiv.com). 

[Guandao Yang](http://www.guandaoyang.com), 
[Tianyi Zhang](https://scholar.google.com/citations?hl=en&view_op=list_works&gmla=AJsN-F5oL2dqrt5Dli21O3seTVse8viKdodY4EQrZp8EV0BUpG5s1brVEPMWVunGQizs0Lltdmn5cPooQHA77vDxymqIITnUUL-GRlYglybFcTnDURbvEss&user=OI0HSa0AAAAJ#), 
Polina Kirichenko, Junwen Bai, 
[Andrew Gordon Wilson](https://people.orie.cornell.edu/andrew/), 
[Christopher De Sa](http://www.cs.cornell.edu/~cdesa/)

## Introduction

Low precision operations can provide scalability, memory savings, portability, and energy efficiency. 
This paper proposes SWALP, an approach to low precision training that averages low-precision SGD iterates with a modified learning rate schedule. 
SWALP is easy to implement and can match the performance of *full-precision* SGD even with all numbers quantized down to 8 bits, including the gradient accumulators.
Additionally, we show that SWALP converges arbitrarily close to the optimal solution for quadratic objectives, and to a noise ball asymptotically smaller than low precision SGD in strongly convex settings. 

This repo contains the codes to replicate our experiment for CIFAR datasets with VGG16 and PreResNet164. 

## Citing this Work
Please cite our work if you find this approach useful in your research:
```latex
@article{SWALP,
  title={SWALP: Stochastic Weight Averaging for Low-Precision Training},
  author={placehodler},
  journal={placeholder},
  year={2018}
}
```

## Dependencies
* CUDA 9.0
* [PyTorch](http://pytorch.org/) version 1.0
* [torchvision](https://github.com/pytorch/vision/)
* [tensorflow](https://www.tensorflow.org/) to use tensorboard

To install other requirements through `$ pip install -r requirements.txt`.

## Usage

We provide scripts to run Small-block Block Floating Point experiments on CIFAR10 and CIFAR100 with VGG16 or PreResNet164.
Following are scripts to reproduce experimental results.

```bash
bash exp/block_vgg_swa.sh CIFAR10     # SWALP training on VGG16 with Small-block BFP in CIFAR10
bash exp/block_vgg_swa.sh CIFAR100    # SWALP training on VGG16 with Small-block BFP in CIFAR100
bash exp/block_resnet_swa.sh CIFAR10  # SWALP training on PreResNet164 with Small-block BFP in CIFAR10
bash exp/block_resnet_swa.sh CIFAR100 # SWALP training on PreResNet164 with Small-block BFP in CIFAR100
```

TODO : example training diagram

## Results

The low-precision results (SGD-LP and SWALP) are produced by running the scripts in `/exp` folder.
The full-precision results (SGD-FP and SWA-FP) are produced by running the SWA repo.

| Datset   | Model        | SGD-FP     | SWA-FP     | SGD-LP     | SWALP      |
|----------|--------------|------------|------------|------------|------------|
| CIFAR10  | VGG16        | 6.81±0.09  | 6.51±0.14  | 7.61±0.15  | 6.70±0.12  |
|          | PreResNet164 | 4.63±0.18  | 4.03±0.10  |            |            |
| CIFAR100 | VGG16        | 27.23±0.17 | 25.93±0.21 | 29.59±0.32 | 26.65±0.29 |
|          | PreResNet164 | 22.20±0.57 | 19.95±0.19 |            |            |

## References
We use the [SWA repo](https://github.com/timgaripov/swa/) as starter template.
Network architecture implementations are adapted from:
* VGG: [github.com/pytorch/vision/](https://github.com/pytorch/vision/)
* PreResNet: [github.com/bearpaw/pytorch-classification](https://github.com/bearpaw/pytorch-classification)
