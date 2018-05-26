from PIL import Image
import os
import glob
import numpy as np
import zlib
import imageio
import nibabel
import  cv2
import xml.etree.ElementTree as ET


data_dir="/media/milton/ssd1/dataset/competitions/wider_face_detection_2018"


def get_all_train_files():
    train_images=[]
    for file_path in glob.glob(os.path.join(data_dir,"WIDER_train","images/**/**")):
        train_images.append(file_path)
    print("Total Train: {}".format(len(train_images)))

    return


def get_all_validation_files():
    valid_images=[]
    for file_path in glob.glob(os.path.join(data_dir,"WIDER_val","images/**/**")):
        valid_images.append(file_path)
    print("Total validation: {}".format(len(valid_images)))

    return



get_all_train_files()
get_all_validation_files()


