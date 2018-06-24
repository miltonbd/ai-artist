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
