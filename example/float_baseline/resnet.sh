python train.py --dataset CIFAR10 \
       --data_path ./data \
       --model PreResNet110 \
       --epochs=150 \
       --lr_init=0.1 \
       --wd=3e-4 \
       --name float_example/resnet \
       --seed 100 \
       --batch_size 128 \
       --float
