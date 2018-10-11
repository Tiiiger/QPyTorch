python train.py --dataset CIFAR10 --data_path ./data --model VGG16 --epochs=200 --lr_init=0.05 --wd=5e-4 --name example --wl-weight 8 --wl-grad 8 --wl-momentum 8 --wl-activate 8 --wl-error 8 --seed 100 --batch_size 128 
--weight-rounding nearest --activate-rounding nearest --grad-rounding nearest --error-rounding nearest --momentum-rounding nearest --auto_low
