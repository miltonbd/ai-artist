import torchvision
import torch
from torchvision import transforms
import numpy as np
import torch.nn as nn
import models

height=112
num_classes=10
batch_size=4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='/media/milton/ssd1/dataset/cifar/cifar_pytorch', train=True,
                                        download=True, transform=transform)
# trainset=torch.randn(10,3,height,height)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='/media/milton/ssd1/dataset/cifar/cifar_pytorch', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
model=models.resnet152(True, avg_pool_kernel=2)
# output_shape=model.features(torch.randn(batch_size,3,height,width)).shape

# model.classifier=nn.Sequential(
#             nn.Linear(output_shape[1]*output_shape[2]*output_shape[3], 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, num_classes),
#         )

model.to(device)

for batch_idx, (inputs,targets) in enumerate(trainloader):
    # if batch_idx > 10:
    #     break
    inputs.contiguous()
    targets.contiguous()
    # if not self.augment_images==None:
    #     inputs=torch.from_numpy(self.augment_images(inputs.numpy()))
    inputs, targets = inputs.to(device), targets.to(device)
    model(inputs)
    print(inputs.size(0))
