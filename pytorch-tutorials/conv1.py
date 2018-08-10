import torchvision
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np


data=torch.randn(10,3,512,512)
conv1=nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=True)

print(conv1(data).shape)

avgpool1=nn.AvgPool2d(7, stride=1, padding=3)
print( avgpool1(data).shape )