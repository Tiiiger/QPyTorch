import torch

class ModelParamAccumulator(torch.nn.Module):
    """A wrapper class to save the real value parameter accumulator of a model
    """
    def __init__(self, model, weight_quantizer=lambda x: x):
        super(ModelParamAccumulator, self).__init__()
        self.model = model
        self.weight_quantizer = weight_quantizer
        self.pointer_to_params = {}
        self.real_acc_params = {}
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                tmp = m.weight.data.clone()
                self.real_acc_params.append(tmp)
                self.pointer_to_params.append(m.weight)

    def quant_weight(self):
        self.save_real_params()
        for index, param in enumerate(range(self.pointer_to_params)):
            param.data.copy_(self.weight_quantizer(param).data)

    def save_real_params(self):
        """Call this function before quantizing the params
        """
        for index, param in enumerate(range(self.pointer_to_params)):
            self.real_acc_params[index].data.copy_(param.data)

    def restore_real_params(self):
        """Call this function to convert current model's param to real value
        """
        for index, param in enumerate(range(self.real_acc_params)):
            self.pointer_to_params[index].data.copy_(param.data)