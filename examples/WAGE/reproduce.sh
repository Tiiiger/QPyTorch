#! /bin/bash

python train.py \
       --dataset CIFAR10 \
       --data_path $1 \
       --dir ./checkpoint/wage-qtorch/stochastic-GE \
       --model VGG7LP \
       --epochs=300 \
       --wl-weight 2 \
       --weight-rounding nearest \
       --wl-grad 8 \
       --grad-rounding stochastic \
       --wl-activate 8 \
       --activate-rounding nearest \
       --wl-error 8 \
       --error-rounding stochastic \
       --wl-rand 16 \
       --seed 100 \
       --batch_size 128 \
       --qtorch;
