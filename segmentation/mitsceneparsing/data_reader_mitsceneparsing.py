import numpy as np
import imageio
from scipy import misc
import os
import _pickle
from keras.utils import to_categorical
from PIL import Image
import numpy as np
import PIL
import Augmentor


class DataReaderMitSceneParsing(object):

    def __init__(self):
        self.data_dir = "/home/milton/dataset/segmentation/mitsceneparsing"
        self.train_dir = os.path.join(self.data_dir, "ADEChallengeData2016", "images", "training")
        self.validation_dir = os.path.join(self.data_dir, "ADEChallengeData2016", "images", "validation")
        self.test_dir = os.path.join(self.data_dir, "ADEChallengeData2016", "images", "testing")
        self.train_masks_dir = os.path.join(self.data_dir, "ADEChallengeData2016", "annotations", "training")
        self.validation_masks_dir = os.path.join(self.data_dir, "ADEChallengeData2016", "annotations", "validation")
        self.num_channels = 3
        self.image_height = 224
        self.image_width = 224
        self.num_classes = 151
        self.num_threads = 8
        self.resize_train_dir = os.path.join(self.data_dir, "ADEChallengeData2016", "images", "224", "training")
        self.resize_train_masks_dir = os.path.join(self.data_dir, "ADEChallengeData2016", "annotations", "224",
                                                   "training")


    def resize_images(self, image_size):
        if not os.path.exists(self.resize_train_dir):
            os.makedirs(self.resize_train_dir)

        if not os.path.exists(self.resize_train_masks_dir):
            os.makedirs(self.resize_train_masks_dir)

            # for file_name in os.listdir(self.train_dir):
            #     file_path = os.path.join(self.train_dir, file_name)
            #     save_file_path = os.path.join(self.resize_train_dir, file_name)
            #
            #     if os.path.exists(file_path):
            #         #print(file_path);
            #         image=Image.open(file_path)
            #         image_resize = image.resize((224, 224), Image.ANTIALIAS)
            #         #print(save_file_path)
            #         image_resize.save(save_file_path)


        print("resizing mask.")
        #https://stackoverflow.com/questions/23135552/resize-ground-truth-images-without-changing-the-labels
        for file_name in os.listdir(self.train_masks_dir):
            mask_file = os.path.join(self.train_masks_dir, os.path.basename(file_name).split(".")[0] + ".png")
            mask_save_file_path = os.path.join(self.resize_train_masks_dir, os.path.basename(file_name).split(".")[0] + ".jpg")

            # augmentor can not use diffrent mask from image
            if os.path.exists(mask_file):
                #print(mask_file)
                image_mask=Image.open(mask_file)
                #image__maskresize = image_mask.resize((224, 224))
                # print(save_file_path)
                #print(np.unique(image__maskresize))
                #print( np.unique(image_mask))
                #assert np.sum(np.unique(image__maskresize) == np.sum(np.unique(image_mask)))
                image_mask.save(mask_save_file_path)

        print("ended")



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
    obj=DataReaderMitSceneParsing()
    #obj.resize_images(224)
    p = Augmentor.Pipeline(obj.train_dir)
    ##Point to a directory containing ground truth data.
    ##Images with the same file names will be added as ground truth data
    ##and augmented in parallel to the original data.
    p.ground_truth(obj.resize_train_masks_dir)
    # Add operations to the pipeline as normal:
    p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
    p.flip_left_right(probability=0.5)
    p.zoom_random(probability=0.5, percentage_area=0.8)
    p.flip_top_bottom(probability=0.5)
    p.sample(50)

    #obj.resize_images(224)
    # train_images , train_masks = obj.get_validation_files()
    # mask=Image.open(train_masks[10])
    # print("max pixel mask {}".format(np.unique(mask)))
    # mask_resized=mask.resize((224,224))
    # print(np.asarray(mask_resized))
    # print("max pixel mask {}".format(np.unique(mask_resized)))
    #
    # print("{},{}".format(len(train_images), len(train_masks)))




