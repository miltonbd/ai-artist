import torch
from torch import nn

data=torch.randn(20,3,1000,1000)
m=nn.AvgPool2d(4)
print(m(data).size())