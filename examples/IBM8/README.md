# IBM8.pytorch

QPyTorch simulation for the NeurIPS 2018 paper, [Training Deep Neural Networks with 8-bit Floating
Point Numbers](https://papers.nips.cc/paper/7994-training-deep-neural-networks-with-8-bit-floating-point-numbers.pdf).
Note that we simulate the numerical behavior of using the proposed 8-bit and
16-bit floating point number but not the chunk-based accumulation. Accumulation is still done in single precision. Also, 
due to the absence of an official reference, the hyperparameters used in this repo are different from the paper.

## Citation

```bash
./example.sh
```

## Citation
If you find this simulation useful, please cite the original paper,

```bash
@incollection{NIPS2018_7994,
title = {Training Deep Neural Networks with 8-bit Floating Point Numbers},
author = {Wang, Naigang and Choi, Jungwook and Brand, Daniel and Chen, Chia-Yu and Gopalakrishnan, Kailash},
booktitle = {Advances in Neural Information Processing Systems 31},
editor = {S. Bengio and H. Wallach and H. Larochelle and K. Grauman and N. Cesa-Bianchi and R. Garnett},
pages = {7675--7684},
year = {2018},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/7994-training-deep-neural-networks-with-8-bit-floating-point-numbers.pdf}
}
```
