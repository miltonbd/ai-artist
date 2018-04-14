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

class QueueRunnerHelper(object):
    def __init__(self, image_height, image_width,num_classes,num_threads):
        self.image_height=image_height
        self.image_width=image_width
        self.num_classes=num_classes
        self.num_threads = num_threads
        return
    # convert string into tensors
    def process_batch(self,train_input_queue):
        file_content = tf.read_file(train_input_queue[0])
        train_image = tf.image.decode_jpeg(file_content, channels=3)
        resized_image = tf.image.resize_images(train_image, [self.image_height, self.image_width])
        resized_image.set_shape([self.image_height, self.image_width, 3])
        float_image = tf.image.per_image_standardization(resized_image)

        train_label_raw = train_input_queue[1]
        train_label_one_hot = tf.one_hot(train_label_raw, self.num_classes)
        train_label = tf.reshape(train_label_one_hot, [self.num_classes, ])
        return float_image, train_label

    def make_queue(self,float_image, train_label, batch_size):
        q = tf.FIFOQueue(capacity=5 * batch_size, dtypes=[tf.float32, tf.float32],
                         shapes=[(self.image_height, self.image_width, 3), (self.num_classes,)])
        enqueue_op = q.enqueue([float_image, train_label])
        qr = tf.train.QueueRunner(q, [enqueue_op] * self.num_threads)
        tf.train.add_queue_runner(qr)
        batch_data, batch_label = q.dequeue_many(n=batch_size)
        return batch_data, batch_label

    def init_queue(self, data_paths, labels):
        train_images = ops.convert_to_tensor(data_paths, dtype=dtypes.string)
        train_labels = ops.convert_to_tensor(labels, dtype=dtypes.int32)
        # create input queues
        train_input_queue = tf.train.slice_input_producer(
            [train_images, train_labels],
            shuffle=True)
        return  train_input_queue
