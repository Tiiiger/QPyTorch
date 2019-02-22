# Training and Inference with Integers in Deep Neural Networks

QPyTorch implementation for the ICLR 2018 paper, [WAGE](https://arxiv.org/abs/1802.04680). This is replicate from the Tensorflow [repo](https://github.com/boluoweifenda/WAGE) by the paper's authors.
We modify this implementation based on our previous [pytorch implementation](https://github.com/stevenygd/WAGE.pytorch).

With QPyTorch the simulation overhead is much smaller. Results are obtained on a GTX 1080 Ti
| Seeting         | Training Time per epoch | Simulation Overhead |
| -------------   |           ------------- |                     |
| No Quantization |                   13.60 |                   0 |
| QPyTorch        |                   17.50 |                 3.9 |
| PyTorch         |                   21.16 |          7.56(1.9x) |

## Prerequisites
- NVIDIA GPU + CUDA + CuDNN
- PyTorch
- QPyTorch
- TensorboardX 
- Tabulate

Please follow the official instruction to install PyTorch and NVIDIA related prerequisites. Other things should be handled by
```bash
pip install -r requirements.txt
```

## Train
Start training using the following scripts:
```bash
./reproduce.sh
```

## Citation
If you find this paper or this repository helpful, please cite the original paper:
```bash
@inproceedings{
wu2018training,
title={Training and Inference with Integers in Deep Neural Networks},
author={Shuang Wu and Guoqi Li and Feng Chen and Luping Shi},
booktitle={International Conference on Learning Representations},
year={2018},
url={https://openreview.net/forum?id=HJGXzmspb},
} 
```
