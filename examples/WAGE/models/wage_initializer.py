import torch
import math
import numpy as np

def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

def scale_limit(limit, bits_W, name, scale_dict):
    beta = 1.5
    Wm = beta / (2**(bits_W-1))
    scale = 2 ** round(np.log2(Wm/limit))
    scale = max(scale, 1.0)
    limit = max(Wm, limit)
    scale_dict[name] = scale
    return limit

def wage_init_(tensor, bits_W, name, scale_dict, factor=2.0, mode="fan_in"):
    if mode != "fan_in":
        raise NotImplementedError("support only wage normal")

    dimensions = tensor.ndimension()
    if dimensions < 2: raise ValueError("tensor at least is 2d")
    elif dimensions == 2: fan_in = tensor.size(1)
    elif dimensions > 2:
        num_input_fmaps = tensor.size(1)
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
    float_limit = math.sqrt(3 * factor / fan_in)
    quant_limit = scale_limit(float_limit, bits_W, name, scale_dict)
    tensor.data.uniform_(-quant_limit, quant_limit)
