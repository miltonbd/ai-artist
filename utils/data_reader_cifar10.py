import numpy as np
import imageio
from scipy import misc
import os
import _pickle
from keras.utils import to_categorical


data_dir = "/home/milton/dataset/cifar/cifar10"
tran_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
ValidImageFormats= {'jpg','jpeg','png','gif'}


cifar10_class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def create_cifar10_class_dir():
    for class_name in cifar10_class_labels:
        class_dir = os.path.join(tran_dir, class_name)
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)

def cifar10_save():

    # batch_file = os.path.join(data_dir,"batches.meta")
    # with open(batch_file, mode='rb') as f:
    #     classes=_pickle.load(f, encoding="bytes")
    #     class_names=classes[b'label_names']
    #     for i in range(10):
    #         class_names[i] = class_names[i].decode('utf-8')
    #     print(class_names)


    images=[]
    labels=[]
    for i in np.arange(1, 6):
        train_file = os.path.join(data_dir, 'data_batch_{}'.format(i))
        #print(train_file)
        with open(train_file, mode='rb') as f:
            data_dict = _pickle.load(f, encoding="bytes")
            labels_batch = data_dict[b'labels']
            data_batch = data_dict[b'data']
            for j in range(len(labels_batch)):
                images.append(data_batch[j])
                labels.append(labels_batch[j])

    for i in range(len(images)):
        img_flat = images[i]
        img_R = img_flat[0:1024].reshape((32, 32))
        img_G = img_flat[1024:2048].reshape((32, 32))
        img_B = img_flat[2048:3072].reshape((32, 32))
        img = np.dstack((img_R, img_G, img_B))
        #images_final = np.reshape(img, [32 * 32 * 3])
        images_final = img.astype(np.uint8)
        label_index = labels[i]
        label = cifar10_class_labels[label_index]
        file_path = os.path.join(tran_dir,label,"{}.jpg".format(i+1))
        imageio.imwrite(file_path, images_final)
        #print(label)

def cifar10_save_test():

    test_images=[]
    test_labels=[]
    test_file = os.path.join(data_dir, 'test_batch')
    with open(test_file, mode='rb') as f:
        data_dict = _pickle.load(f, encoding="bytes")
        labels_batch = data_dict[b'labels']
        data_batch = data_dict[b'data']
        for j in range(len(labels_batch)):
            test_images.append(data_batch[j])
            test_labels.append(labels_batch[j])
    for i in range(len(test_images)):
        img_flat = test_images[i]
        img_R = img_flat[0:1024].reshape((32, 32))
        img_G = img_flat[1024:2048].reshape((32, 32))
        img_B = img_flat[2048:3072].reshape((32, 32))
        img = np.dstack((img_R, img_G, img_B))
        #images_final = np.reshape(img, [32 * 32 * 3])
        images_final = img.astype(np.uint8)
        label_index = test_labels[i]
        label = cifar10_class_labels[label_index]
        file_path = os.path.join(test_dir,label,"{}.jpg".format(i+1))
        imageio.imwrite(file_path, images_final)


#cifar10_save()


def get_test_files_cifar_10_classification():
   files=[]
   labels=[]
   for class_name in cifar10_class_labels:
       class_dir = os.path.join(test_dir, class_name)
       for file_name in os.listdir(class_dir):
           file_path = os.path.join(class_dir, file_name)
           files.append(file_path)
           labels.append(cifar10_class_labels.index(class_name))
   print("path:{}, label:{}".format(files[0],labels[0]))
   return files, labels


def get_train_files_cifar_10_classification():
   train_files=[]
   train_labels=[]
   for class_name in cifar10_class_labels:
       class_dir = os.path.join(tran_dir, class_name)
       for file_name in os.listdir(class_dir):
           file_path = os.path.join(class_dir, file_name)
           train_files.append(file_path)
           train_labels.append(cifar10_class_labels.index(class_name))
   return train_files, train_labels


class DataReaderCifar10(object):
    """
    data_dir = "/home/milton/dataset/cifar/cifar10" # contains data_batch_{1..5}
    """
    def __init__(self, batch_size, gpu_nums):
        self.data_dir =  "/home/milton/dataset/cifar/cifar10"
        self.batch_size = batch_size
        self.epoch = 0
        self.itr = 0
        self.itr_test = 0
        self.total_train_count = 0
        self.gpu_nums = gpu_nums
        print("batch, gpu nums".format(batch_size, gpu_nums))
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
        self.total_test_count = np.minimum(len(self.images_test), len(self.labels_test))

        print("cifar10 loaded with {} test items".format(len(self.labels_test)))

        #print(self.labels[0])


    def nextBatchTrain(self):
        if self.itr >= self.total_train_count:
            self.itr = 0
            self.epoch += 1
        #print("index in data {}".format(self.itr))

        batch_size_gpus = np.min([self.batch_size * self.gpu_nums, self.total_train_count-self.itr])
        # There may be lower number of values
        # than batch at the end of epoch.
        #print("batch {}".format(batch_size_gpus))
        images_final = np.zeros(shape=[batch_size_gpus, 32*32*3],dtype=np.uint8)
        labels_final = np.zeros(shape=[batch_size_gpus,], dtype=np.int)


        for i in range(batch_size_gpus):
            img_flat = self.images[self.itr]
            img_R = img_flat[0:1024].reshape((32, 32))
            img_G = img_flat[1024:2048].reshape((32, 32))
            img_B = img_flat[2048:3072].reshape((32, 32))
            img = np.dstack((img_R, img_G, img_B))
            images_final[i] = np.reshape(img,[32*32*3])
            labels_final[i] = self.labels[self.itr]
            self.itr+=1
        images_final = images_final/255
        #print(images_final.shape)
        return (images_final, to_categorical(labels_final,num_classes=10))

    def nextTestBatch(self):
        # if self.itr >= self.total_test_count:
        #     self.itr = 0
        #     self.epoch += 1
        batch_size = self.batch_size * self.gpu_nums
        images_final = np.zeros(shape=[batch_size, 32*32*3], dtype=np.uint8)
        labels_final = np.zeros(shape=[batch_size, ], dtype=np.int)

        for i in range(batch_size):
            img_flat = self.images_test[self.itr_test]
            img_R = img_flat[0:1024].reshape((32, 32))
            img_G = img_flat[1024:2048].reshape((32, 32))
            img_B = img_flat[2048:3072].reshape((32, 32))
            img = np.dstack((img_R, img_G, img_B))
            images_final[i] = np.reshape(img,[32*32*3])
            labels_final[i] = self.labels_test[self.itr_test]
            self.itr_test += 1
        images_final = images_final/255
        return (images_final, to_categorical(labels_final, num_classes=10))



    def shuffle(self):
        pass



if __name__ == '__main__':
    create_cifar10_class_dir()
    cifar10_save_test()