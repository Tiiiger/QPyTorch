# QPyTorch Low-Precision Training Tutorial

In this directory, we provide a code base for general low-precision training. You can train
VGGs and ResNets with fixed points, block floating points, and floating points.

## Instructions & Commands

See available options by
```bash
python train.py -h
```

Train a VGG16 with low-precision weight and gradients in fixed points by
```bash
bash sample_scripts/fixed_baseline.sh
```

Train a PreResNet20 with google bfloat16 end to end by
```bash
bash sample_scripts/bfloat_baseline.sh
```

Train a VGG16 with google bfloat16 end to end by
```bash
bash sample_scripts/float_baseline/bfloat_baseline.sh
```

## File Structure
- `train.py` contains the high-level training code and usage of QPyTorch
- `utils.py` contains useful helpers that can be shared with low-precision and full-precision training
- `data.py` defines datasets
- `models/` contains model definitions. `*_low.py` are the low-precision model counterparts.
