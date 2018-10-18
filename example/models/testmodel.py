import torch.nn as nn
import torchvision.transforms as transforms

__all__ = ['LinearLP']

class SimpleLinearLP(nn.Module):
    def __init__(self, quant, num_classes=10):
        super(LinearLP, self).__init__()
        self.classifier = nn.Linear(3072, num_classes)
        self.quant = quant()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.quant(x)
        return x

class Base:
    base = SimpleLinearLP
    args = list()
    kwargs = dict()
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


class LinearLP(Base):
    pass