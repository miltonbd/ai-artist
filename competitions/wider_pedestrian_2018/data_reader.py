from PIL import Image
import os
import glob
import numpy as np
import zlib
import imageio
import nibabel
import  cv2
import xml.etree.ElementTree as ET


data_dir="/media/milton/ssd1/dataset/competitions/wider_pedestrian_detection_2018"


def get_all_train_files():
    train_images=[]
    for file_path in glob.glob(os.path.join(data_dir,"train","**")):
        train_images.append(file_path)
    print("Total train: {}".format(len(train_images)))
    return


def get_all_validation_files():
    validation_images = []
    for file_path in glob.glob(os.path.join(data_dir, "val", "**")):
        validation_images.append(file_path)
    print("Total valid: {}".format(len(validation_images)))

    return


get_all_train_files()
get_all_validation_files()
