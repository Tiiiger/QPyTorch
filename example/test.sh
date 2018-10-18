CUDA_VISIBLE_DEVICES=1 python train.py -cfg 20_LAYER_16 --epoch 100  --ckpt_dir 20_model_test_fix_epoch;
CUDA_VISIBLE_DEVICES=1 python train.py -cfg 20_LAYER_32 --epoch 100  --ckpt_dir 20_model_test_fix_epoch;
CUDA_VISIBLE_DEVICES=1 python train.py -cfg 20_LAYER_48 --epoch 100  --ckpt_dir 20_model_test_fix_epoch;
CUDA_VISIBLE_DEVICES=1 python train.py -cfg 20_LAYER_64 --epoch 100  --ckpt_dir 20_model_test_fix_epoch;
CUDA_VISIBLE_DEVICES=1 python train.py -cfg 20_LAYER_80 --epoch 100  --ckpt_dir 20_model_test_fix_epoch;
CUDA_VISIBLE_DEVICES=1 python train.py -cfg 20_LAYER_96 --epoch 100  --ckpt_dir 20_model_test_fix_epoch;
CUDA_VISIBLE_DEVICES=1 python train.py -cfg 20_LAYER_112 --epoch 100 --ckpt_dir 20_model_test_fix_epoch;
CUDA_VISIBLE_DEVICES=1 python train.py -cfg 20_LAYER_128 --epoch 100 --ckpt_dir 20_model_test_fix_epoch;
CUDA_VISIBLE_DEVICES=1 python train.py -cfg 20_LAYER_144 --epoch 100 --ckpt_dir 20_model_test_fix_epoch;
CUDA_VISIBLE_DEVICES=1 python train.py -cfg 20_LAYER_160 --epoch 100 --ckpt_dir 20_model_test_fix_epoch;


CUDA_VISIBLE_DEVICES=1 python attack.py --ckpt_dir ./20_model_test_fix_epoch/vgg_20_LAYER_16/ ;
CUDA_VISIBLE_DEVICES=1 python attack.py --ckpt_dir ./20_model_test_fix_epoch/vgg_20_LAYER_32/ ;
CUDA_VISIBLE_DEVICES=1 python attack.py --ckpt_dir ./20_model_test_fix_epoch/vgg_20_LAYER_48/ ;
CUDA_VISIBLE_DEVICES=1 python attack.py --ckpt_dir ./20_model_test_fix_epoch/vgg_20_LAYER_64/ ;
CUDA_VISIBLE_DEVICES=1 python attack.py --ckpt_dir ./20_model_test_fix_epoch/vgg_20_LAYER_80/ ;
CUDA_VISIBLE_DEVICES=1 python attack.py --ckpt_dir ./20_model_test_fix_epoch/vgg_20_LAYER_96/ ;
CUDA_VISIBLE_DEVICES=1 python attack.py --ckpt_dir ./20_model_test_fix_epoch/vgg_20_LAYER_112/ ;
CUDA_VISIBLE_DEVICES=1 python attack.py --ckpt_dir ./20_model_test_fix_epoch/vgg_20_LAYER_128/ ;
CUDA_VISIBLE_DEVICES=1 python attack.py --ckpt_dir ./20_model_test_fix_epoch/vgg_20_LAYER_144/ ;
CUDA_VISIBLE_DEVICES=1 python attack.py --ckpt_dir ./20_model_test_fix_epoch/vgg_20_LAYER_160/ ;