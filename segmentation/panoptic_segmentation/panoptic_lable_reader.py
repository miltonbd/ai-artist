import glob
import os
import scipy.misc as m
import numpy as np

lbl_path=glob.glob(os.path.join('/media/milton/ssd1/dataset/mscoco/panoptic_annotations_trainval2017/annotations/panoptic_val2017/panoptic_val2017','**'))[0]
print(lbl_path)

img=m.imread(lbl_path)

