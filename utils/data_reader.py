import numpy as np
import imageio
import os


ValidImageFormats= {'jpg','jpeg','png','gif'}
VGG_MEAN = [103.939, 116.779, 123.68]# Mean value of pixels in R G and B channels


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
        images_batch = []
        labels_batch = []
        for index in range(itr * self.batch_size, itr * self.batch_size + self.batch_size ):
            image_path = self.images[index]
            label_path = self.labels[index]

            image = imageio.imread(image_path)

            label = imageio.imread(label_path)

            R, G , B , A = image

            R = R - VGG_MEAN[0]
            G = G - VGG_MEAN[1]
            B = B - VGG_MEAN[2]

            image = np.asarray([R,G,B,A])

            images_batch.append(image)
            labels_batch.append(label)


        return (images_batch, labels_batch)

    def shuffle(self):
        pass

