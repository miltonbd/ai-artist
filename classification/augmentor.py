import glob
import os
import Augmentor
import shutil
import os

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

def random_crop(pipeline, n):
    pipeline.crop_random(probability=1, percentage_area=.5)
    pipeline.sample(n)

def random_crop_center(pipeline, n):
    pipeline.crop_centre(probability=1, percentage_area=.5)
    pipeline.sample(n)

def random_zoom(pipeline, n):
    pipeline.zoom_random(probability=1, percentage_area=.5)
    pipeline.sample(n)


root_directory = "/home/milton/dataset/skin/classification_train_224/*"
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
for folder in folders:
    print("Folder %s:" % (folder))
    pipeline = (Augmentor.Pipeline(folder))
    n=len(pipeline.augmentor_images)
    skew_corner(pipeline, n/2)
    skew_left_right(pipeline, n/2)
    flip_left_right(pipeline, n/2)
    skew_top_bottom(pipeline, n/2)
    shear(pipeline, n/2)
    rotate270(pipeline, n/2)
    skew_tilt(pipeline, n/2)
    rotate(pipeline, n/)
    random_distortion(pipeline, n/2)
    flip_top_bottom(pipeline, n/2)
    rotate90(pipeline, n/2)
    flip_random(pipeline,n/2)
    random_erasing(pipeline, n)
    print("\n----------------------------\n")




print("Augmentation ended.")
