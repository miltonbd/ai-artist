import glob
import os
import Augmentor
import shutil
import os
from PIL import Image

def skew_corner(pipeline, n):
    pipeline.skew_corner(probability=1, magnitude=.2)
    pipeline.sample(n)

def skew_left_right(pipeline, n):
    pipeline.skew_left_right(probability=1, magnitude=.2)
    pipeline.sample(n)

def flip_left_right(pipeline, n):
    pipeline.flip_left_right(probability=1)
    pipeline.sample(n)

def skew_top_bottom(pipeline, n):
    pipeline.skew_top_bottom(probability=1, magnitude=.2)
    pipeline.sample(n)

def shear(pipeline, n):
    pipeline.shear(probability=1, max_shear_left=5, max_shear_right=5)
    pipeline.sample(n)

def rotate270(pipeline, n):
    pipeline.rotate270(probability=1)
    pipeline.sample(n)

def skew_tilt(pipeline, n):
    pipeline.skew_tilt(probability=1, magnitude=.2)
    pipeline.sample(n)

def rotate(pipeline, n):
    pipeline.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
    pipeline.sample(n)

def random_distortion(pipeline, n):
    pipeline.random_distortion(probability=1, grid_width=16, grid_height=16, magnitude=16)
    pipeline.sample(n)

def flip_top_bottom(pipeline, n):
    pipeline.flip_top_bottom(probability=1)
    pipeline.sample(n)

def rotate180(pipeline, n):
    pipeline.rotate180(probability=1)
    pipeline.sample(n)

def rotate90(pipeline, n):
    pipeline.rotate90(probability=1)
    pipeline.sample(n)

def flip_random(pipeline, n):
    pipeline.flip_random(probability=1)
    pipeline.sample(n)

def histo_gram_equa(pipeline, n):
    pipeline.histogram_equalisation(probability=1)
    pipeline.sample(n)

def random_erasing(pipeline, n):
    pipeline.random_erasing(probability=1, rectangle_area=.2)
    pipeline.sample(n)

def resize(pipeline, n):
    pipeline.random_erasing(probability=1, rectangle_area=.2)
    pipeline.sample(n)

root_directory = "/home/milton/dataset/segmentation/carvana/train/*"
mask_dir="/home/milton/dataset/segmentation/carvana/masks/"
gif_mask_dir="/home/milton/dataset/segmentation/carvana/train_masks/*"

import numpy as np
def convert_gif_jpg():
    for f in glob.glob(gif_mask_dir):
        name=f.split('_mask.gif')[0].split('/')[-1]
        neW_mask_file = os.path.join(mask_dir,name+".jpg")
        img=Image.open(f)
        imgnp = np.asarray(img)*255
        #print(imgnp.shape)
        out=Image.fromarray(imgnp)
        out.save(neW_mask_file)
        #break

#convert_gif_jpg()

def resize_train_masks():

    # for f in glob.glob(root_directory):
    #     img = np.asarray(Image.open(f))
    #     img_resized=Image.fromarray(img)
    #     img_resized=img_resized.resize((320,320),Image.NORMAL)
    #     #assert np.unique(img_resized).all()==np.unique(img).all()
    #     name = f.split('_mask.gif')[0].split('/')[-1]
    #     neW_mask_file = os.path.join("/home/milton/dataset/segmentation/carvana/train_372", name)
    #     img_resized.save(neW_mask_file)
    # print("resizing masks")
    for f in glob.glob(gif_mask_dir):
        img = np.asarray(Image.open(f))*255
        img_resized=Image.fromarray(img)
        img_resized=img_resized.resize((320,320),Image.NORMAL)
        assert np.unique(img_resized).all()==np.unique(img).all()
        name = f.split('_mask.gif')[0].split('/')[-1]
        neW_mask_file = os.path.join("/home/milton/dataset/segmentation/carvana/train_masks_372", name + ".jpg")

        img_resized.save(neW_mask_file)







resize_train_masks()
exit(1)

folders = []
for f in glob.glob(root_directory):
    if os.path.isdir(f):
        folders.append(os.path.abspath(f))
        old_path=os.path.join(f,'output')
        shutil.rmtree(old_path)
        os.makedirs(old_path)
        
print("Folders (classes) found: %s " % [os.path.split(x)[1] for x in folders])

# for p in pipelines.values():
#     print("Class %s has %s samples." % (p.augmentor_images[0].class_label, len(p.augmentor_images)))

pipelines = {}
# for folder in folders:
#     print("Folder %s:" % (folder))
#     pipeline = (Augmentor.Pipeline(folder))
#     n=len(pipeline.augmentor_images)
#     skew_corner(pipeline, n)
#     skew_left_right(pipeline, n)
#     flip_left_right(pipeline, n)
#     skew_top_bottom(pipeline, n)
#     shear(pipeline, n)
#     rotate270(pipeline, n)
#     skew_tilt(pipeline, n)
#     rotate(pipeline, n)
#     random_distortion(pipeline, n)
#     flip_top_bottom(pipeline, n)
#     rotate180(pipeline, n)
#     rotate90(pipeline, n)
#     flip_random(pipeline,n)
#     histo_gram_equa(pipeline, n)
#     random_erasing(pipeline, n)
#     print("\n----------------------------\n")




print("Augmentation ended.")
