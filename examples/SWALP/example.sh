#! /bin/bash

DATA=CIFAR100
MODEL=VGG16
SEED=100
python3 train.py \
    --dataset ${DATA} \
    --data_path . \
    --model ${MODEL} \
    --epochs=300 \
    --lr_init=0.05 \
    --swa_start 200 \
    --swa_lr 0.01 \
    --wd=5e-4 \
    --seed ${SEED} \
    --wl-weight 8 \
    --wl-grad 8 \
    --wl-activate 8 \
    --wl-error 8 \
    --wl-momentum 8 \
    --rounding stochastic;
