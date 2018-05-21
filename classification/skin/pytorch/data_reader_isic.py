from torch.utils.data.dataset import Dataset

from classification.skin.data_loader_isic import DataReaderISIC2017
import numpy as np
import imageio
import torch
from torchvision import transforms
import Augmentor


class ISIC2017Dataset(Dataset):
    """
    task1 =melanoma
    task2=sebrorreheic
    mode= train, validation, test
    """
    def __init__(self, task=1, mode='train'):
        self.task=task
        loader = DataReaderISIC2017(batch_size=10, epochs=1, gpu_nums=1)
        loader.loadDataSet()
        self.melanoma_train=np.asarray(loader.getTrainDataForClassificationMelanoma())
        self.sebkeratosis_train=np.asarray(loader.getTrainDataForClassificationSeborrheicKeratosis())
        self.melanoma_validation=np.asarray(loader.getValidationDataForClassificationMelanoma())
        self.melanoma_test=np.asarray(loader.getTestDataForClassificationMelanoma())
        self.mode = mode
        self.transforms = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((123.68, 116.779, 103.939), (1,1,1))])

    def __getitem__(self, index):

        if self.task==1 :
            if self.mode == 'train':
                train_x, train_y, labels = self.melanoma_train
                img = imageio.imread(train_x[index])
                img=img.astype(np.float32)
                data = self.transforms(img)
                label = train_y[index]
                return (data, np.argmax(label))
            elif self.mode == 'validation':
                valid_x, valid_y, labels = self.melanoma_validation
                return valid_x[index], valid_y[index]
            else:
                test_x, test_y,labels = self.melanoma_test
                img = imageio.imread(test_x[index])
                img = img.astype(np.float32)
                data = self.transforms(img)
                label = test_y[index]
                return (data, np.argmax(label))


    def __len__(self):

        if self.task == 1 :
            if self.mode == 'train':
                train_x, train_y, labels = self.melanoma_train
                return len(train_x)
            elif self.mode == 'validation':
                valid_x, valid_y, labels = self.melanoma_validation
                return len(valid_x)
            else:
                test_x, test_y,labels = self.melanoma_test
                return len(test_x)

