import torch
from qtorch.quant import *

if __name__ == "__main__":
    wl, fl = 16, 16
    a = torch.linspace(- 2 ** (wl-fl-1), 2 ** (wl-fl-1) - 2 ** (-fl), steps=100, device='cuda')
    b = torch.zeros_like(a)
    num_of_trails = 100
    for i in range(num_of_trails):
        b = b * i/(i+1.) + fixed_point_quantize(a, forward_wl=wl, forward_fl=fl, backward_wl=wl, backward_fl=fl,
                                                forward_rounding="stochastic", backward_rounding="stochastic") / (i+1)
    print(torch.nn.L1Loss()(a,b))
    print(torch.nn.MSELoss()(a,b))

    # quant = Quantizer(16, 16, 16, 16, "nearest", "nearest", "fixed", "fixed")
    # from example import 
