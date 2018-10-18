import torch
from torch import nn
from qtorch.quant import *
from qtorch.auto_low import *

class SimpleLinearLP(nn.Module):
    def __init__(self, quant, size=3072):
        super(SimpleLinearLP, self).__init__()
        self.classifier = nn.Linear(size, size)
        self.classifier.weight.data = torch.ones_like(self.classifier.weight.data)
        self.classifier.bias.data = torch.ones_like(self.classifier.bias.data)
        self.quant = quant

    def forward(self, x):
        x = x.view(1, x.size(0))
        for i in range(10):
            x = self.classifier(x)
            x = self.quant(x)
        return x

class SimpleLinear(nn.Module):
    def __init__(self, size=3072):
        super(SimpleLinear, self).__init__()
        self.classifier = nn.Linear(size, size)
        self.classifier.weight.data = torch.ones_like(self.classifier.weight.data)
        self.classifier.bias.data = torch.ones_like(self.classifier.bias.data)

    def forward(self, x):
        x = x.view(1, x.size(0))
        for i in range(10):
            x = self.classifier(x)
        return x

class SimpleConv(nn.Module):
    def __init__(self, num_channels=10, kernel_size=3):
        super(SimpleConv, self).__init__()
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size, padding=1)
        self.conv1.weight.data = torch.ones_like(self.conv1.weight.data)
        self.conv1.bias.data = torch.ones_like(self.conv1.bias.data)

        self.classifier = nn.Conv2d(num_channels, num_channels, kernel_size, padding=1)
        self.classifier.weight.data = torch.ones_like(self.classifier.weight.data)
        self.classifier.bias.data = torch.ones_like(self.classifier.bias.data)

    def forward(self, x):
        x = self.conv1(x)
        for i in range(10):
            x = self.classifier(x)
        return x

class SimpleConvLP(nn.Module):
    def __init__(self, quant, num_channels=10, kernel_size=3):
        super(SimpleConvLP, self).__init__()
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size, padding=1)
        self.conv1.weight.data = torch.ones_like(self.conv1.weight.data)
        self.conv1.bias.data = torch.ones_like(self.conv1.bias.data)
        self.quant = quant
        self.classifier = nn.Conv2d(num_channels, num_channels, kernel_size, padding=1)
        self.classifier.weight.data = torch.ones_like(self.classifier.weight.data)
        self.classifier.bias.data = torch.ones_like(self.classifier.bias.data)

    def forward(self, x):
        x = self.conv1(x)
        x = self.quant(x)
        for i in range(10):
            x = self.classifier(x)
            x = self.quant(x)
        return x

if __name__ == "__main__":
    wl, fl = 16, 16
    rounding_type = "nearest"
    number_type = "fixed"
    device = "cuda"
    a = torch.linspace(- 2 ** (wl-fl-1), 2 ** (wl-fl-1) - 2 ** (-fl), steps=3072, device=device)
    print(a.size())
    # a_copy = a.clone()
    a_copy = a
    quant = Quantizer(16, 16, 16, 16, rounding_type, rounding_type, number_type, number_type)
    
    test_conv = False
    if test_conv:
        conv_channel = 512
        import math
        width = int(math.sqrt(a.size(0) / 3))
        a = a.reshape(1,3, width,width)
        a_copy = a.clone()
        print(f"Test convolutional layer with {conv_channel} channels")
        model_1 = SimpleConvLP(quant, num_channels=conv_channel).to(device)
        model_2 = SimpleConv(num_channels=conv_channel).to(device)
        model_3 = SimpleConv(num_channels=conv_channel).to(device)
        lower(model_2, 
              layer_types=["conv"],
              wl_activate=wl, 
              wl_error=wl,
              fl_activate=fl, 
              fl_error=fl,
              activate_rounding=rounding_type,
              error_rounding=rounding_type,
              activate_type=number_type,
              error_type=number_type
        )
    else:
        matrix_size = 2048
        a = torch.linspace(- 2 ** (wl-fl-1), 2 ** (wl-fl-1) - 2 ** (-fl), steps=matrix_size, device=device)
        a_copy = a.clone()
        print(f"Test linear layer with size {matrix_size} x {matrix_size}")
        model_1 = SimpleLinearLP(quant, size=matrix_size).to(device)
        model_2 = SimpleLinear(size=matrix_size).to(device)
        model_3 = SimpleLinear(size=matrix_size).to(device)
        lower(model_2, 
              layer_types=["linear"],
              wl_activate=wl, 
              wl_error=wl,
              fl_activate=fl, 
              fl_error=fl,
              activate_rounding=rounding_type,
              error_rounding=rounding_type,
              activate_type=number_type,
              error_type=number_type
        )
    import time
    output_1 = model_1(a)
    output_1 = model_1(a)
    output_1 = model_1(a)
    loop = 1000
    total_time = 0
    for i in range(loop):
        time_now = time.time()
        output_1 = model_1(a)
        time_since_1 = time.time() - time_now
        total_time += time_since_1

    output_2 = model_2(a_copy)
    output_2 = model_2(a_copy)
    output_2 = model_2(a_copy)
    total_time_2 = 0
    for i in range(loop):
        time_now = time.time()
        output_2 = model_2(a_copy)
        time_since_2 = time.time() - time_now
        total_time_2 += time_since_2

    output_3 = model_3(a_copy)
    output_3 = model_3(a_copy)
    output_3 = model_3(a_copy)
    total_time_3 = 0
    for i in range(loop):
        time_now = time.time()
        output_3 = model_3(a_copy)
        time_since_3 = time.time() - time_now
        total_time_3 += time_since_3

    # print(torch.nn.L1Loss()(a,a_copy))
    # print(torch.nn.L1Loss()(output_1,output_2))
    # print(torch.nn.L1Loss()(output_1,output_3))
    print(f"Running time - Model with manually inserted LP layers: {total_time}")
    print(f"Running time - Model with auto inserted LP layers: {total_time_2}")
    print(f"Running time - Model with no LP layers: {total_time_3}")

