import numpy as np
import imageio
from scipy import misc
import os
import _pickle
import math


class DataReaderISIC2017(object):
    """
    data_dir = "/home/milton/dataset/cifar/cifar10" # contains data_batch_{1..5}
    """
    def __init__(self, batch_size, epochs, gpu_nums):
        self.data_dir =  "/home/milton/dataset/skin/"

        self.batch_size = batch_size
        self.epoch = 0
        self.itr = 0
        self.total_train_count = 0
        self.epochs = epochs
        self.gpu_nums = gpu_nums

        # cifar is smaller dataset which can be hold in ram, for bigger dataset the dataset should be loaded from storage while
        # building next batch.
        self.images_train = []
        self.labels_train = []
        self.images_valid = []
        self.labels_valid = []
        self.images_test = []
        self.labels_test = []

        return

    def initIterationsCount(self):
        self.iterations = int( math.ceil(self.total_train_count / (self.batch_size * self.gpu_nums)))


    def getMelanoma(self, images_dir):
        melanomas_dir = os.path.join(self.data_dir, images_dir, 'melanomas')
        seborrheic_keratosis_dir = os.path.join(self.data_dir, images_dir, 'seborrheic_keratosis')
        nevus_dir = os.path.join(self.data_dir, images_dir, 'nevus')

        melanomas = np.array([])
        for name in os.listdir(melanomas_dir):
            path = os.path.join(melanomas_dir, name)
            melanomas = np.append(melanomas, path)
        nonmelanomas = np.array([])

        for name in os.listdir(seborrheic_keratosis_dir):
            path = os.path.join(seborrheic_keratosis_dir, name)
            nonmelanomas = np.append(nonmelanomas, path)

        for name in os.listdir(nevus_dir):
            path = os.path.join(nevus_dir, name)
            nonmelanomas = np.append(nonmelanomas, path)

        #print("{} melanomas {}".format(images_dir, len(melanomas)))
        #print("{} nonmelanomas {}".format(images_dir, len(nonmelanomas)))

        return (melanomas, nonmelanomas)

    def getSeborrheic(self, images_dir):
        melanomas_dir = os.path.join(self.data_dir, images_dir, 'melanomas')
        seborrheic_keratosis_dir = os.path.join(self.data_dir, images_dir, 'seborrheic_keratosis')
        nevus_dir = os.path.join(self.data_dir, images_dir, 'nevus')

        seborrheic_keratosis = np.array([])

        for name in os.listdir(seborrheic_keratosis_dir):
            path = os.path.join(seborrheic_keratosis_dir, name)
            seborrheic_keratosis = np.append(seborrheic_keratosis, path)

        nonseborrheic_keratosis = np.array([])
        for name in os.listdir(melanomas_dir):
            path = os.path.join(melanomas_dir, name)
            nonseborrheic_keratosis = np.append(nonseborrheic_keratosis, path)

        for name in os.listdir(nevus_dir):
            path = os.path.join(nevus_dir, name)
            nonseborrheic_keratosis = np.append(nonseborrheic_keratosis, path)

        #print("{} Seborrheic {}".format(images_dir, len(seborrheic_keratosis)))
        #print("{} NonSeborrheic {}".format(images_dir, len(nonseborrheic_keratosis)))

        return (seborrheic_keratosis, nonseborrheic_keratosis)

    def loadDataSet(self):

        #load for classification Task 1: melanona vs seb or nevus

        self.melanomas_train, self.nonmelanomas_train = self.getMelanoma('classification_train_224')
        self.seborrheic_keratosis_train, self.nonseborrheic_keratosis_train = self.getSeborrheic('classification_train_224')

        #self.total_train_count = np.minimum(len(self.images), len(self.labels))
        #self.iterations = int(self.epochs * self.total_train_count / (self.batch_size * self.gpu_nums))

        # loading validation

        self.melanomas_valid, self.nonmelanomas_valid = self.getMelanoma('classification_valid_224')
        self.seborrheic_keratosis_valid, self.nonseborrheic_keratosis_valid = self.getSeborrheic(
            'classification_valid_224')

        # loading test
        self.melanomas_test, self.nonmelanomas_test = self.getMelanoma('classification_test_224')
        self.seborrheic_keratosis_test, self.nonseborrheic_keratosis_test = self.getSeborrheic('classification_test_224')

    def getValidationDataForClassificationMelanoma(self):
        train_paths = np.concatenate((self.melanomas_valid, self.nonmelanomas_valid), axis=0)
        train_x = []
        for train_path in train_paths:
            X = imageio.imread(train_path).reshape(224 * 224 * 3) / 255
            train_x.append(X)
        train_y = np.zeros(shape=[len(train_x), 2], dtype=np.float32)
        train_y[0:len(self.melanomas_valid), 0] = 1
        train_y[len(self.nonmelanomas_valid):, 1] = 1
        labels = ['Melanoma', 'Non Melanoma']
        #print(len(train_x))
        #print(len(train_y))
        train_x = np.asarray(train_x)
        #print(train_x.shape)
        self.initIterationsCount()
        return train_x, train_y, labels


    def getTrainDataForClassificationMelanoma(self):
        train_paths = np.concatenate((self.melanomas_train , self.nonmelanomas_train), axis=0)
        train_x = []
        for train_path in train_paths:
            X = imageio.imread(train_path).reshape(224 * 224 * 3)/255
            train_x.append(X)
        train_y = np.zeros(shape=[len(train_x),2],dtype=np.float32)
        train_y[0:len(self.melanomas_train),0]=1
        train_y[len(self.melanomas_train):,1]=1
        labels=['Melanoma','Non Melanoma']
        print("Total train items for melanona {}".format(len(train_x)))
        train_x = np.asarray(train_x)
        #print(train_x.shape)
        self.total_train_count = len(train_x)
        self.initIterationsCount()

        return train_x, train_y, labels

    def getTestDataForClassificationMelanoma(self):
        test_paths = np.concatenate((self.melanomas_test, self.nonmelanomas_test), axis=0)
        test_x = []
        for test_path in test_paths:
            X = imageio.imread(test_path).reshape(224 * 224 * 3) / 255
            test_x.append(X)
        test_y = np.zeros(shape=[len(test_x), 2], dtype=np.float32)
        test_y[0:len(self.melanomas_test), 0] = 1
        test_y[len(self.melanomas_test):, 1] = 1
        labels = ['Melanoma', 'Non Melanoma']
        #print(len(test_x))
        #print(len(test_y))
        test_x = np.asarray(test_x)
        #print(test_x.shape)
        self.initIterationsCount()
        return test_x, test_y, labels

    def getDataForClassificationSeborrheicKeratosis(self):
        return (self.seborrheic_keratosis, self.nonseborrheic_keratosis,['Seborrheic Keratosis','Non Seborrheic Keratosis'])


    def nextBatch(self):
        if self.itr >= self.total_train_count:
            self.itr = 0
            self.epoch += 1
        #print("index in data {}".format(self.itr))
        batch_size = np.min([self.batch_size * self.gpu_nums, self.total_train_count-self.itr])
        # There may be lower number of values
        # than batch at the end of epoch.

        images_final = np.zeros(shape=[batch_size, 32, 32, 3],dtype=np.uint8)
        labels_final = np.zeros(shape=[batch_size, ], dtype=np.int)


        for i in range(batch_size):
            img_flat = self.images[self.itr]
            img_R = img_flat[0:1024].reshape((32, 32))
            img_G = img_flat[1024:2048].reshape((32, 32))
            img_B = img_flat[2048:3072].reshape((32, 32))
            img = np.dstack((img_R, img_G, img_B))
            images_final[i] = np.reshape(img,[32,32,3])
            labels_final[i] = self.labels[self.itr]
            self.itr+=1
        images_final = images_final/255
        return (images_final, labels_final)

    def testBatch(self):
        batch_size = self.batch_size
        images_final = np.zeros(shape=[batch_size, 32, 32, 3], dtype=np.uint8)
        labels_final = np.zeros(shape=[batch_size, ], dtype=np.int)

        for i in range(batch_size):
            img_flat = self.images_test[self.itr]
            img_R = img_flat[0:1024].reshape((32, 32))
            img_G = img_flat[1024:2048].reshape((32, 32))
            img_B = img_flat[2048:3072].reshape((32, 32))
            img = np.dstack((img_R, img_G, img_B))
            images_final[i] = np.reshape(img, [32, 32, 3])
            labels_final[i] = self.labels_test[self.itr]
            self.itr += 1
        images_final = images_final/255
        return (images_final, labels_final)

    def shuffle(self):
        pass



if __name__ == '__main__':
    #data_reader = DataReaderISIC2017(100,10,2)
    #data_reader.loadDataSet()
    import tensorflow as tf
    print(tf.VERSION)