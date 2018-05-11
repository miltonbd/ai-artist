import os
import skimage.io as io
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import imageio


data_dir = '/home/milton/dataset/segmentation/BRATS/BRATS2015/'
train_dir_hgg = os.path.join(data_dir, "training","HGG")
train_dir_lgg = os.path.join(data_dir, "training", "LGG")

path = '/home/milton/dataset/segmentation/BRATS/BRATS2015/training/HGG/brats_2013_pat0001_1/VSD.Brain.XX.O.MR_T1.54513.mha'
path2='/home/milton/dataset/segmentation/BRATS/MICCAI_BraTS17_Data_Training/HGG/Brats17_2013_2_1/Brats17_2013_2_1_seg.nii.gz'
label_path="/home/milton/dataset/segmentation/BRATS/BRATS2015/training/HGG/brats_2013_pat0001_1/VSD.Brain_3more.XX.O.OT.54517.mha"

img= sitk.ReadImage(label_path)
images=[]
for i in range(155):
    slices = sitk.GetArrayViewFromImage(img)[i,:,:]
    if np.unique(slices).sum()>0:
        imageio.imsave('images/{}_label.png'.format(i), slices)
        images.append(slices)

imageio.mimsave("label.gif",images)


