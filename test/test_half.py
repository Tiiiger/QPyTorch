import torch
from qtorch.quant import *

if __name__ == "__main__":
    a = torch.rand(int(1e4)).cuda()
    man = 11
    exp = 5
    half_a = a.half()
    def compute_dist(a, half_a):
        diff =  (a-half_a.float())
        diff2 = diff**2
        return diff2.sum()
    sim_half_a = float_quantize(a, forward_man_bits=man, forward_exp_bits=exp, forward_rounding="nearest")
    assert compute_dist(a, sim_half_a) == compute_dist(a, half_a), "half float test [failed]"
    print("half float test [passed]")
    #sim_half_a = float_quantize(a, forward_man_bits=man, forward_exp_bits=exp, forward_rounding="stochastic")
    #print("stochastic simulate single half difference {}".format(compute_dist(a, sim_half_a)))
