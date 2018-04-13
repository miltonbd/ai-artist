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

data_dir = "/home/milton/dataset/pascal/VOCdevkit/VOC2012/ImageSets/Main"
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
NUM_CHANNELS = 3
BATCH_SIZE = 5
num_threads=6

voc_class_labels = ['aeroplane','bicycle', 'bird', 'boat',
                    'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                    'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                    'train', 'tvmonitor']

def get_all_files_voc_2012_classification():
    jpeg_dir="/home/milton/dataset/pascal/VOCdevkit/VOC2012/JPEGImages"
    files=[]
    labels=[]
    for label_name in voc_class_labels:
        train_file_pascal_2012 = os.path.join(data_dir,"{}_train.txt".format(label_name))
        with open(train_file_pascal_2012) as f:
            files_class = [os.path.join(jpeg_dir,line.rstrip().split(" ")[0]+".jpg")   for line in f.readlines()]
            files+=files_class
            labels+=[voc_class_labels.index(label_name) for _ in files_class]
    #print("Pascal voc 2012 train images: {}, {}".format(len(files),len(labels)))
    return files, labels



# convert string into tensors

all_filepath_labels=get_all_files_voc_2012_classification()

#random.shuffle(all_filepath_labels)
all_filepaths, all_labels = all_filepath_labels
train_images = ops.convert_to_tensor(all_filepaths, dtype=dtypes.string)
train_labels = ops.convert_to_tensor(all_labels, dtype=dtypes.int32)

# create input queues
train_input_queue = tf.train.slice_input_producer(
    [train_images, train_labels],
    shuffle=True)

file_content = tf.read_file(train_input_queue[0])
train_image = tf.image.decode_jpeg(file_content, channels=3)
resized_image = tf.image.resize_images(train_image, [IMAGE_HEIGHT, IMAGE_WIDTH])
resized_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, 3])
float_image = tf.image.per_image_standardization(resized_image)

q = tf.FIFOQueue(3, tf.float32)
enqueue_op = q.enqueue_many(float_image)
threads = 6
qr = tf.train.QueueRunner(q, [enqueue_op] * threads)
tf.train.add_queue_runner(qr)
data = q.dequeue()

print("input pipeline ready")

with tf.Session() as sess:
    # initialize the variables
    sess.run(tf.global_variables_initializer())

    # initialize the queue threads to start to shovel data
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    print ("from the train set:")
    for i in range(20):
        print(sess.run(data))

    coord.request_stop()
    coord.join(threads)
    sess.close()