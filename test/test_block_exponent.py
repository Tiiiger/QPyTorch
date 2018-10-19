import torch
from qtorch.quant import *

if __name__ == "__main__":
    # block floating point should perform the same with fixed point when the max exponent is 0
    wl = 5
    fl = 4
    a = torch.linspace(-0.9, 0.9, steps=20, device='cuda')
    fixed_a = fixed_point_quantize(a, forward_wl=wl, forward_fl=fl, forward_rounding='nearest')
    block_a = block_quantize(a, forward_wl=wl, forward_rounding='nearest')
    assert torch.eq(fixed_a, block_a).all(), "test [failed]"
    print("test [passed]")
