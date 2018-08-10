import torch
import torch.nn as nn

data=torch.randn(30, 3, 400, 400)
data.cuda()

print(data)

