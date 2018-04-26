from torch.utils.data.dataset import Dataset

from classification.skin.data_loader_isic import DataReaderISIC2017
import numpy as np
import imageio
import torch
import torchvision
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
        p = Augmentor.Pipeline("/home/milton/dataset/skin/classification_train_224/images")
        # Point to a directory containing ground truth data.
        # Images with the same file names will be added as ground truth data
        # and augmented in parallel to the original data.
        #p.ground_truth("/path/to/ground_truth_images")
        # Add operations to the pipeline as normal:
        p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
        p.flip_left_right(probability=0.5)
        p.zoom_random(probability=0.5, percentage_area=0.8)
        p.flip_top_bottom(probability=0.5)
        #
        self.transform = torchvision.transforms.Compose([
            p.torch_transform(),
            torchvision.transforms.ToTensor(),
        ])
        # self.transform = transforms.Compose(
        #     [transforms.ToTensor(),
        #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, index):

        if self.task==1 :
            if self.mode == 'train':
                train_x, train_y, labels = self.melanoma_train
                img = imageio.imread(train_x[index])
                img=img.astype(np.float32)
                data = self.transform(img)
                label = train_y[index]
                return (data, np.argmax(label))
            elif self.mode == 'validation':
                valid_x, valid_y, labels = self.melanoma_validation
                return valid_x[index], valid_y[index]
            else:
                test_x, test_y,labels = self.melanoma_test
                img = imageio.imread(test_x[index]).reshape([3, 224, 224]) / 255
                img = img.astype(np.float32)
                data = torch.from_numpy(img.copy())
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

