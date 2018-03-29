import numpy as np
import imageio
from scipy import misc
import os
import _pickle


class DataReaderISIC2017(object):
    """
    data_dir = "/home/milton/dataset/cifar/cifar10" # contains data_batch_{1..5}
    """
    def __init__(self, batch_size, epochs, gpu_nums):
        self.data_dir =  "/home/milton/dataset/cifar/cifar10"
        self.batch_size = batch_size
        self.epoch = 0
        self.itr = 0
        self.total_train_count = 0
        self.epochs = epochs
        self.gpu_nums = gpu_nums

        # cifar is smaller dataset which can be hold in ram, for bigger dataset the dataset should be loaded from storage while
        # building next batch.
        self.images = []
        self.labels = []
        self.images_test = []
        self.labels_test = []
        return

    def loadDataSet(self):
        for i in np.arange(1, 6):
            train_file = os.path.join(self.data_dir, 'data_batch_{}'.format(i))
            with open(train_file, mode='rb') as f:
                data_dict = _pickle.load(f, encoding="bytes")
                labels_batch = data_dict[b'labels']
                data_batch = data_dict[b'data']
                for j in labels_batch:
                    self.images.append(data_batch[j])
                    self.labels.append(labels_batch[j])
            train_file = os.path.join(self.data_dir, 'test_batch')
        with open(train_file, mode='rb') as f:
            data_dict = _pickle.load(f, encoding="bytes")
            labels_batch = data_dict[b'labels']
            data_batch = data_dict[b'data']
            batch_size = self.batch_size
            for j in labels_batch:
                self.images_test.append(data_batch[j])
                self.labels_test.append(labels_batch[j])

        # for i in self.labels:
        #     print("{}".format(self.labels[i]))
        print("cifar10 loaded with {} train items".format(len(self.labels)))
        self.total_train_count = np.minimum(len(self.images), len(self.labels))
        self.total_train_count_test = np.minimum(len(self.images_test), len(self.labels_test))
        self.iterations = int(self.epochs * self.total_train_count / (self.batch_size * self.gpu_nums))
        print("Total train itrations needed {}".format(self.iterations))
        self.iterations_test = int(self.total_train_count_test / self.batch_size )

        print("cifar10 loaded with {} test items".format(len(self.labels_test)))
        print("Total test itrations needed {}".format(self.iterations_test))



        #print(self.labels[0])


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

