__all__ = ["PositTanhModule","PositTanhModuleEnhanced","RefTanhModule"]
#Todo : implement sigmoid, rarely used in modern DNN
import torch
from qtorch.quant import posit_sigmoid, posit_tanh, posit_tanh_enhanced
class PositTanhModule(torch.nn.Module):
    def forward(self, input):
        return PositTanhFunction.apply(input)
        #return torch.tanh(input)
    
class PositTanhFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        # replace posit_tanh_enhanced <> posit_tanh for different approx
        output = posit_tanh(input,nsize=16,es=0)
        ctx.save_for_backward(output)
        return output#input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        #tanhx = top_data[i];
        #bottom_diff[i] = top_diff[i] * (1 - tanhx * tanhx);
        #grad_input = grad_output.clone()
        grad_input = grad_output*(1- output*output)
        return grad_input
    
class PositTanhModuleEnhanced(torch.nn.Module):
    def forward(self, input):
        return PositTanhFunctionEnhanced.apply(input)
    
class PositTanhFunctionEnhanced(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        output = posit_tanh_enhanced(input,nsize=16,es=0)
        ctx.save_for_backward(output)
        return output#input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        grad_input = grad_output*(1- output*output)
        return grad_input
    

    
class RefTanhModule(torch.nn.Module):
    def forward(self, input):
        return torch.tanh(input)
    