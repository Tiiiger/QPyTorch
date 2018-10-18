import torch
from torch import nn
from qtorch.quant import *

__all__ =  ['lower']

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

def lower(model,
          layer_types=[],
          wl_activate=-1,
          wl_error=-1,
          fl_activate=-1,
          fl_error=-1,
          activate_man=-1,
          error_man=-1,
          activate_exp=-1,
          error_exp=-1,
          activate_rounding=None,
          error_rounding=None,
          activate_type=None,
          error_type=None):
    quant = Quantizer(wl_activate, wl_error,
                      fl_activate, fl_error,
                      activate_man,
                      error_man,
                      activate_exp,
                      error_exp,
                      activate_rounding, error_rounding,
                      activate_type, error_type)
    lower_func = _get_apply_lower_func(quant, layer_types=layer_types)
    model.apply(lower_func)



