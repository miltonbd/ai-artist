import glob
from PIL import Image
import numpy as np

pre_encoded_dir='/media/milton/ssd1/dataset/pascal/VOCdevkit/VOC2012/SegmentationClass/pre_encoded'
pascal_mask='/media/milton/ssd1/dataset/pascal/VOCdevkit/VOC2012/SegmentationClass'

img=Image.open(glob.glob(pre_encoded_dir+"/**")[10])
print(np.unique(img))

img1=Image.open(glob.glob(pascal_mask+"/**")[10])
print(np.unique(img1))

print(np.asarray(img))
print(np.asarray(img1))
