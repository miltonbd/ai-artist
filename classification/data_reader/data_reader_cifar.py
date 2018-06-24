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

cifar_pytorch="/media/milton/ssd1/dataset/cifar/cifar_pytorh"
trainset = torchvision.datasets.CIFAR10(root=cifar_pytorch, train=True,
                                        download=True, transform=[])
trainset = torchvision.datasets.CIFAR100(root=cifar_pytorch, train=True,
                                        download=True, transform=[])
data_dir="/media/milton/ssd1/dataset/cifar"

cifar_10_batches=os.path.join(data_dir,"cifar-10-batches-py")

def get_data(files):
    for file in files:
        with open(file, 'rb') as fo:
            data= _pickle.load(fo, encoding='bytes')
            print(data[0])


def get_train_data():
    files=glob.glob(os.path.join(cifar_10_batches,'data_batch_*'))
    return get_data(files)

def get_validation_data():
    files=glob.glob(os.path.join(data_dir, 'Validation_256/**/**.jpg'))
    return get_data(files)

class DatasetReader(Dataset):
    """
    """
    def __init__(self, data,mode='train',):
        print("{} count:{}".format(mode,len(data)))
        self.mode=mode
        self.data=np.asarray(data)
        self.transform_train_image=transforms.Compose([
            transforms.RandomCrop([224,224]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()]);
        self.transform_test_image = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor()]);


    def __getitem__(self, index):
        img_path=self.data[index,0]
        label=int(self.data[index,1])

        if not os.path.exists(img_path):
            print("{} image not found".format(img_path))
            exit(0);
        img = Image.open(img_path)
        if self.mode=="train":
            data = self.transform_train_image(img)
            return data, label

        elif self.mode=="valid":
            data = self.transform_test_image(img)
            return data, label

    def __len__(self):
        return len(self.data)

def get_data_sets(batch_size1, batch_size2):
    train_data_set = DatasetReader(get_train_data(),"train")
    validation_data_set = DatasetReader(get_validation_data(),"valid")
    trainloader = torch.utils.data.DataLoader(train_data_set, batch_size=batch_size1, shuffle=True,
                                              num_workers=2)
    valloader = torch.utils.data.DataLoader(validation_data_set, batch_size=batch_size2, shuffle=False,
                                              num_workers=2)
    return (trainloader, valloader)

def test():
    trainloader, valloader = get_data_sets(100)
    for idx, (inputs, targets) in enumerate(valloader):
        print(inputs.shape)

if __name__ == '__main__':
    get_train_data()