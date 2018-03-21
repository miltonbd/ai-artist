import numpy as np
import imageio
from scipy import misc
import os


ValidImageFormats= {'jpg','jpeg','png','gif'}


class DataReader(object):
    def __init__(self, image_dir, label_dir, batch_size):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.batch_size = batch_size
        self.epoch = 0
        self.itr = 0
        self.total_train_count = 0
        return

    def loadDataSet(self):
        image_names = os.listdir(self.image_dir)
        self.images=[]
        for img in image_names:
            image_path = os.path.join(self.image_dir, img)
            self.images.append(image_path)

        print("Total train images {}".format(len(self.images)))

        label_names = os.listdir(self.label_dir)
        self.labels=[]
        for label in label_names:
            label_path = os.path.join(self.label_dir, label)
            self.labels.append(label_path)

        print("Total train labels {}".format(len(self.labels)))

        self.total_train_count = np.minimum(len(self.images), len(self.labels))



    def nextBatch(self, itr):
        if self.itr >= self.total_train_count:
            self.itr = 0
            self.epoch += 1

        batch_size = np.min([self.batch_size, self.total_train_count-self.itr]) # There may be lower numbe rof values
        # than batch at the end of epoch.

        images_batch = []
        labels_batch = []
        for _ in range( batch_size ):
            image_path = self.images[self.itr]
            label_path = self.labels[self.itr]

            image = imageio.imread(image_path)
            image = image[:, :, 0:3]
            label = imageio.imread(label_path)
            label_h, label_w = label.shape
            image = misc.imresize(image,[label_h, label_w])
            images_batch.append(image)
            labels_batch.append(label)

        # find the lowest height, width in segmenttaion image and resize the
        label_min_h = 0
        label_min_w = 0

        for i in range(len(labels_batch)):
            h, w = labels_batch[i].shape
            if i == 0:
                label_min_h = h
                label_min_w = w
            else:
                if h < label_min_h and w < label_min_w :
                    label_min_h = h
                    label_min_w = w

        #print("Batch label (h,w) is ({},{}) ".format(label_min_h, label_min_w))

        images_final = np.zeros(shape=[batch_size, label_min_h, label_min_w, 3],dtype=np.int)
        labels_final = np.zeros(shape=[batch_size, label_min_h, label_min_w, 1], dtype=np.int)

        for i in range(batch_size):

            images_final[i] = misc.imresize(images_batch[i],[label_min_h, label_min_w], interp="bilinear")
            la = misc.imresize(labels_batch[i],[label_min_h, label_min_w], interp="nearest")
            la = la.reshape((label_min_h, label_min_w, 1))
            labels_final[i] = la

        return (images_final, labels_final)


    def shuffle(self):
        pass

