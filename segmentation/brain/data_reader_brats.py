import numpy as np
import imageio
from scipy import misc
import os
import _pickle
from keras.utils import to_categorical
import PIL
from PIL import Image
import numpy as np
import SimpleITK as sitk

class DataReaderBrats(object):

    def __init__(self):

        self.data_dir = "/home/milton/dataset/segmentation/BRATS/BRATS2015"
        self.train_dir_hgg = os.path.join(self.data_dir, "ADEChallengeData2016", "images", "training")
        self.validation_dir = os.path.join(self.data_dir, "ADEChallengeData2016", "images", "validation")
        self.test_dir = os.path.join(self.data_dir, "ADEChallengeData2016", "images", "testing")
        self.train_masks_dir = os.path.join(self.data_dir, "ADEChallengeData2016", "annotations", "training")
        self.validation_masks_dir = os.path.join(self.data_dir, "ADEChallengeData2016", "annotations", "validation")
        self.num_channels = 3
        self.image_height = 224
        self.image_width = 224
        self.num_classes = 151
        self.num_threads = 8


    def resize_images(self, image_size):
        self.resize_train_dir = os.path.join(self.data_dir, "ADEChallengeData2016", "images", "224", "training")
        self.resize_train_masks_dir = os.path.join(self.data_dir, "ADEChallengeData2016", "annotations", "224", "training")

        if not os.path.exists(self.resize_train_dir):
            os.makedirs(self.resize_train_dir)

        if not os.path.exists(self.resize_train_masks_dir):
            os.makedirs(self.resize_train_masks_dir)

        for file_name in os.listdir(self.resize_train_dir):
            file_path = os.path.join(self.resize_train_dir, file_name)
            if os.path.exists(file_path):
                image=PIL.Image(file_path)
                image_resize = image.resize((224, 224), Image.ANTIALIAS)
                image_resize.save('sompic.jpg')
            mask_file = os.path.join(self.resize_train_masks_dir, os.path.basename(file_path).split(".")[0] + ".png")
            # print(mask_file)
            if os.path.exists(mask_file):
                train_mask_files.append(mask_file)



    def get_train_files(self):
       train_files=[]
       train_mask_files=[]
       for file_name in os.listdir(self.train_dir):
           file_path = os.path.join(self.train_dir, file_name)
           if os.path.exists(file_path):
               train_files.append(file_path)
           mask_file = os.path.join(self.train_masks_dir, os.path.basename(file_path).split(".")[0]+".png")
           #print(mask_file)
           if os.path.exists(mask_file):
               train_mask_files.append(mask_file)
       return train_files, train_mask_files


    def get_validation_files(self):
       train_files=[]
       train_mask_files=[]
       for file_name in os.listdir(self.validation_dir):
           file_path = os.path.join(self.validation_dir, file_name)
           if os.path.exists(file_path):
               train_files.append(file_path)
           mask_file = os.path.join(self.validation_masks_dir, os.path.basename(file_path).split(".")[0]+".png")
           #print(mask_file)
           if os.path.exists(mask_file):
               train_mask_files.append(mask_file)
       return train_files, train_mask_files


    def get_test_files(self):
       train_files=[]
       train_mask_files=[]
       for file_name in os.listdir(self.test_dir):
           file_path = os.path.join(self.test_dir, file_name)
           print(file_path)
           if os.path.exists(file_path):
               train_files.append(file_path)
       return train_files


if __name__ == '__main__':
    obj=DataReaderBrats()





