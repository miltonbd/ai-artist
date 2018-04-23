import numpy as np
import imageio
from scipy import misc
import os
import _pickle
from keras.utils import to_categorical

data_dir = "/home/milton/dataset/segmentation/carvana"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
masks_dir = os.path.join(data_dir, "train_masks")

image_height=1280
image_width=1918


def get_train_files_carvana_segmentation():
   train_files=[]
   train_mask_files=[]
   for file_name in os.listdir(train_dir):
       file_path = os.path.join(train_dir, file_name)
       train_files.append(file_path)
       mask_file = os.path.join(masks_dir, os.path.basename(file_path),".gif")
       train_mask_files.append(mask_file)
   return train_files, train_mask_files



def get_test_files_carvana_segmentation():
   train_files=[]
   train_mask_files=[]
   for file_name in os.listdir(train_dir):
       file_path = os.path.join(train_dir, file_name)
       train_files.append(file_path)
       mask_file = os.path.join(masks_dir, os.path.basename(file_path),".gif")
       train_mask_files.append(mask_file)
   return train_files, train_mask_files


