from torch.utils.data.dataset import Dataset

from classification.skin.data_loader_isic import DataReaderISIC2017
import numpy as np
import imageio
import torch
from torchvision import transforms
import os


class CarvanaDataset(Dataset):
    """
    mode= train, validation, test
    """
    def __init__(self, mode='train'):
        loader = DataReaderCarvana()
        self.train=np.asarray(loader.get_train_files())
        self.test=np.asarray(loader.get_test_files())
        self.mode = mode
        self.transforms = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, index):

            if self.mode == 'train':
                train_x, train_y = self.train
                img = imageio.imread(train_x[index])
                img=img.astype(np.float32)
                data = self.transforms(img)

                mask = imageio.imread(train_y[index])
                mask_data=torch.from_numpy(mask)
                #data_mask = self.transforms(mask_img)

                return (data, mask_data)
            elif self.mode == 'validation':
                valid_x, valid_y, labels = self.validation
                return valid_x[index], valid_y[index]
            else:
                test_x, test_y = self.test
                img = imageio.imread(test_x[index])
                img = img.astype(np.float32)
                data = self.transforms(img)
                label = test_y[index]
                return (data, np.argmax(label))


    def __len__(self):
        if self.mode == 'train':
            train_x, train_y = self.train
            return len(train_x)
        elif self.mode == 'validation':
            valid_x, valid_y, labels = self.validation
            return len(valid_x)
        else:
            test_x, test_y= self.test
            return len(test_x)


class DataReaderCarvana(object):

    def __init__(self):
        self.data_dir = "/home/milton/dataset/segmentation/carvana"
        self.train_dir = os.path.join(self.data_dir, "train")
        self.test_dir = os.path.join(self.data_dir, "test")
        self.train_masks_dir = os.path.join(self.data_dir, "train_masks")
        self.num_channels = 3
        self.image_height = 1280
        self.image_width = 1918
        self.num_classes = 2
        self.num_threads = 4

    def get_train_files(self):
       train_files=[]
       train_mask_files=[]
       for file_name in os.listdir(self.train_dir):
           file_path = os.path.join(self.train_dir, file_name)
           train_files.append(file_path)
           mask_file = os.path.join(self.train_masks_dir, os.path.basename(file_path).split(".")[0]+"_mask.gif")
           train_mask_files.append(mask_file)
       print("Total train {}".format(len(train_files)))
       return train_files, train_mask_files

    def get_test_files(self):
       test_files=[]
       train_mask_files=[]
       for file_name in os.listdir(self.test_dir):
           file_path = os.path.join(self.test_dir, file_name)
           test_files.append(file_path)
           #mask_file = os.path.join(self.masks_dir, os.path.basename(file_path),".gif")
           #train_mask_files.append(mask_file)
       return test_files, 1





