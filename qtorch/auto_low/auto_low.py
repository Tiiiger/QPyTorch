import torch
from torch import nn
from qtorch.quant import *
from collections import OrderedDict
import copy

__all__ =  ['lower', 'sequential_lower']

SEQUENTIAL_LAYERS = [nn.Sequential, nn.ModuleList] # TODO: Param List

DICT_LAYERS = [nn.ModuleDict]

CONV_LAYERS = [nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d,
               nn.ConvTranspose3d, nn.Unfold, nn.Fold]

POOL_LAYERS = [nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d, nn.MaxUnpool1d, nn.MaxUnpool2d,
               nn.MaxUnpool3d, nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
               nn.FractionalMaxPool2d, nn.LPPool1d, nn.LPPool2d, nn.AdaptiveMaxPool1d,
               nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool2d,
               nn.AdaptiveMaxPool1d, nn.AdaptiveAvgPool1d,
               nn.AdaptiveMaxPool3d, nn.AdaptiveAvgPool3d]

PAD_LAYERS = [nn.ReflectionPad1d, nn.ReflectionPad2d,
              nn.ReplicationPad1d, nn.ReplicationPad2d,
              nn.ZeroPad2d,
              nn.ConstantPad1d, nn.ConstantPad2d, nn.ConstantPad3d]

ACTIVATION_LAYERS = [nn.ELU, nn.Hardshrink, nn.Hardtanh, nn.LeakyReLU, nn.LogSigmoid,
                     nn.PReLU, nn.ReLU, nn.ReLU6, nn.RReLU,
                     nn.SELU, nn.Sigmoid, nn.Softplus, nn.Softshrink,
                     nn.Softsign, nn.Tanh, nn.Tanhshrink, nn.Threshold,
                     nn.Softmin, nn.Softmax, nn.Softmax2d, nn.LogSoftmax,
                    ]#nn.AdaptiveLogSoftmaxWithLoss]

NORM_LAYERS = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
               nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
               nn.LayerNorm, nn.LocalResponseNorm]

# Not supporting RNN layer

LINEAR_LAYERS = [nn.Linear, nn.Bilinear]

DROPOUT_LAYERS = [nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout]

# Not supporting Sparse/Distance layers

LOSS_LAYERS = [nn.L1Loss, nn.MSELoss, nn.CrossEntropyLoss, nn.NLLLoss, nn.PoissonNLLLoss,
               nn.KLDivLoss, nn.BCELoss, nn.BCEWithLogitsLoss, nn.MarginRankingLoss,
               nn.HingeEmbeddingLoss, nn.MultiLabelMarginLoss, nn.SmoothL1Loss,
               nn.SoftMarginLoss, nn.MultiLabelSoftMarginLoss, #nn.CosineEmbeddingLos,
               nn.MultiMarginLoss, nn.TripletMarginLoss]

LAYERS_TYPES = {
                    "conv":CONV_LAYERS,
                    "linear":LINEAR_LAYERS,
                    "pool":POOL_LAYERS,
                    "pad":PAD_LAYERS,
                    "activation":ACTIVATION_LAYERS,
                    "normalization":NORM_LAYERS,
                    "dropout":DROPOUT_LAYERS,
                    "loss":LOSS_LAYERS
               }

def _get_apply_lower_func(quant, layer_types=[]):
    def _insert_LP_layer(module):
        """Insert quant layer for all layers so long as in layer_types
        """
        lp_layer_types = []
        for layer_type in layer_types:
            assert layer_type in LAYERS_TYPES.keys()
            lp_layer_types += LAYERS_TYPES[layer_type]

        old_forward = module.forward
        if type(module) in lp_layer_types:
            module.forward = lambda *input : quant(old_forward(*input))
        else:
            return
    return _insert_LP_layer

def _get_return_sequential_lower_func(quant, layer_types=[]):
    def _insert_LP_layer(module):
        """Insert quant layer for all layers so long as in layer_types
        """
        if type(module) in SEQUENTIAL_LAYERS:
            for i, sub_module in enumerate(module.children()):
                module[i] = _insert_LP_layer(module[i])
            return module
        elif type(module) in DICT_LAYERS:
            for key, sub_module in module.items():
                module[key] = _insert_LP_layer(module[key])
            return module
        elif len(list(module.children())) != 0:
            for attribute_name in module.__dir__():
                attribute = getattr(module, attribute_name)
                if isinstance(attribute, nn.Module):
                    setattr(module, attribute_name, _insert_LP_layer(attribute))
            return module
        else:
            # Module has no submodule
            lp_layer_types = []
            for layer_type in layer_types:
                assert layer_type in LAYERS_TYPES.keys()
                lp_layer_types += LAYERS_TYPES[layer_type]

            if type(module) in lp_layer_types:
                module = nn.Sequential(module, quant)
            return module

    return _insert_LP_layer
 

def lower(model,
          layer_types=[],
          forward_number=None, backward_number=None,
          forward_rounding="stochastic", backward_rounding="stochastic"):
    quant = Quantizer(forward_number, backward_number,
                      forward_rounding, backward_rounding)
    lower_func = _get_apply_lower_func(quant, layer_types=layer_types)
    model.apply(lower_func)


def sequential_lower(model,
                     layer_types=[],
                     forward_number=None, backward_number=None,
                     forward_rounding="stochastic", backward_rounding="stochastic"):
    """Return a new model without touching the old one
    """
    quant = Quantizer(forward_number, backward_number,
                      forward_rounding, backward_rounding)
    
    lower_func = _get_return_sequential_lower_func(quant, layer_types=layer_types)
    return lower_func(copy.deepcopy(model))

# def sequential_lower_(model,
#                       layer_types=[],
#                       forward_number=None, backward_number=None,
#                       forward_rounding="stochastic", backward_rounding="stochastic"):
#     """Modify the original model
#     """
#     quant = Quantizer(forward_number, backward_number,
#                       forward_rounding, backward_rounding)
    
#     lower_func = _get_return_sequential_lower_func(quant, layer_types=layer_types)
#     return _get_return_sequential_lower_func(quant, layer_types=layer_types)(model)



