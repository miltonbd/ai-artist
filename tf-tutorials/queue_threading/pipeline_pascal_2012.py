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

data_dir = "/home/milton/dataset/pascal/VOCdevkit/VOC2012/ImageSets/Main"
image_height = 224
image_width = 224
num_channels = 3
num_classes=10
batch_size = 128
num_threads=8 # keep 4 for 2 gpus

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
        print("Train batch: {}. Test batch: {}".format(feed_batch_data.shape, feed_batch_label.shape))
    coord.request_stop()
    coord.join(threads)
    sess.close()
    print("Time Taken {}".format(time.time()-start))