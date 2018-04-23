import numpy as np
import imageio
from scipy import misc
import os
import _pickle
from keras.utils import to_categorical

import numpy as np
import imageio
from scipy import misc
import os
import _pickle
from keras.utils import to_categorical
import PIL
from PIL import Image

class DataReaderMitSceneParsing(object):

    def __init__(self):
        self.data_dir = "/home/milton/dataset/segmentation/carvana"
        self.train_dir = os.path.join(self.data_dir, "train")
        self.validation_dir = os.path.join(self.data_dir, "test")
        self.test_dir = os.path.join(self.data_dir, "ADEChallengeData2016", "images", "testing")
        self.train_masks_dir = os.path.join(self.data_dir, "train_masks")
        self.validation_masks_dir = os.path.join(self.data_dir, "ADEChallengeData2016", "annotations", "validation")
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
           mask_file = os.path.join(self.masks_dir, os.path.basename(file_path),".gif")
           train_mask_files.append(mask_file)
       return train_files, train_mask_files



    def get_validation_files(self):
       train_files=[]
       train_mask_files=[]
       for file_name in os.listdir(self.train_dir):
           file_path = os.path.join(self.train_dir, file_name)
           train_files.append(file_path)
           mask_file = os.path.join(self.masks_dir, os.path.basename(file_path),".gif")
           train_mask_files.append(mask_file)
       return train_files, train_mask_files


