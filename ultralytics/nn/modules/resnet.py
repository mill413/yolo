# https://blog.csdn.net/u014297502/article/details/128787691
from torchvision import models
import torch.nn as nn
'''
模型: resnet50
'''

__all__ = ('resnet501', 'resnet502', 'resnet503')

class resnet501(nn.Module):
    def __init__(self, ignore) -> None:
        super().__init__()
        model = models.resnet50()
        modules = list(model.children())
        modules = modules[:6]
        self.model = nn.Sequential(*modules)
    def forward(self, x):
        return self.model(x)
    
class resnet502(nn.Module):
    def __init__(self, ignore) -> None:
        super().__init__()
        model = models.resnet50()
        modules = list(model.children())
        modules = modules[6]
        self.model = nn.Sequential(*modules)
    def forward(self, x):
        return self.model(x)
    
class resnet503(nn.Module):
    def __init__(self, ignore) -> None:
        super().__init__()
        model = models.resnet50()
        modules = list(model.children())
        modules = modules[7]
        self.model = nn.Sequential(*modules)
    def forward(self, x):
        return self.model(x)
