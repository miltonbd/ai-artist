from torch.utils.data.dataset import Dataset
import numpy as np
import imageio
import torch
from torchvision import transforms
from statics_isic import *
import glob
from PIL import Image
import os
import threading
import os
import torchvision

cifar10_classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def get_cifar10_dataset_loader(batch_size, transform):
    cifar_pytorch = "/media/milton/ssd1/dataset/cifar/cifar_pytorh"

    trainset = torchvision.datasets.CIFAR10(root=cifar_pytorch, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=cifar_pytorch, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    return (trainloader, testloader)



def get_cifar100_dataset_loader(batch_size, transform):
    cifar_pytorch = "/media/milton/ssd1/dataset/cifar/cifar_pytorh"

    trainset = torchvision.datasets.CIFAR100(root=cifar_pytorch, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root=cifar_pytorch, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    return (trainloader, testloader)



