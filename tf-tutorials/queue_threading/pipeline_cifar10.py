import tensorflow as tf
import time
import os
import imageio
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import tensorflow as tf
import random
import os
import time
import _pickle

data_dir = "/home/milton/dataset/cifar/cifar10"
image_height = 32
image_width = 32
num_channels = 3
num_classes=10
batch_size = 128
num_threads=8 # keep 4 for 2 gpus

voc_class_labels = ['aeroplane','bicycle', 'bird', 'boat',
                    'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                    'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                    'train', 'tvmonitor']
cifar10_class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def cifar10_save():
    # batch_file = os.path.join(data_dir,"batches.meta")
    # with open(batch_file, mode='rb') as f:
    #     classes=_pickle.load(f, encoding="bytes")
    #     class_names=classes[b'label_names']
    #     for i in range(10):
    #         class_names[i] = class_names[i].decode('utf-8')
    #     print(class_names)
    for class_name in cifar10_class_labels:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)

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
        file_path = os.path.join(data_dir,label,"{}.jpg".format(i+1))
        imageio.imwrite(file_path, images_final)
        #print(label)


#cifar10_save()

def get_train_files_cifar_10_classification():
   train_files=[]
   train_labels=[]
   for class_name in cifar10_class_labels:
       class_dir = os.path.join(data_dir, class_name)
       for file_name in os.listdir(class_dir):
           file_path = os.path.join(class_dir, file_name)
           train_files.append(file_path)
           train_labels.append(cifar10_class_labels.index(class_name))
   return (train_files, train_labels)

# convert string into tensors

rain_filepath_labels=get_train_files_cifar_10_classification()

#random.shuffle(all_filepath_labels)
train_filepaths, all_labels = rain_filepath_labels
train_images = ops.convert_to_tensor(train_filepaths, dtype=dtypes.string)
train_labels= ops.convert_to_tensor(all_labels, dtype=dtypes.int32)
# create input queues
train_input_queue = tf.train.slice_input_producer(
    [train_images, train_labels],
    shuffle=True)

file_content = tf.read_file(train_input_queue[0])
train_image = tf.image.decode_jpeg(file_content, channels=3)
resized_image = tf.image.resize_images(train_image, [image_height, image_width])
resized_image.set_shape([image_height, image_width, 3])
float_image = tf.image.per_image_standardization(resized_image)

train_label_raw=train_input_queue[1]
train_label_one_hot = tf.one_hot(train_label_raw, num_classes)
train_label=tf.reshape(train_label_one_hot,[num_classes,])

q = tf.FIFOQueue(capacity=5*batch_size, dtypes=[tf.float32, tf.float32],
                 shapes=[(image_height, image_width, 3), (num_classes,)])
enqueue_op = q.enqueue([float_image,train_label])
qr = tf.train.QueueRunner(q, [enqueue_op] * num_threads)
tf.train.add_queue_runner(qr)
batch_data, batch_label = q.dequeue_many(n=batch_size)

print("input pipeline ready")
start = time.time()
with tf.Session() as sess:
    # initialize the variables
    sess.run(tf.global_variables_initializer())

    # initialize the queue threads to start to shovel data
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    print ("from the train set:")
    for i in range(20):
        feed_batch_data, feed_batch_label = sess.run([batch_data, batch_label])
        #print("Train Data batch: {}. Labels batch: {}".format(feed_batch_data.shape, feed_batch_label.shape))
        #print(feed_batch_label[0])
    coord.request_stop()
    coord.join(threads)
    sess.close()
    print("Time Taken {}".format(time.time()-start))