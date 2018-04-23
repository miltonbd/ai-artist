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

    def make_queue_segmentation(self, float_image, train_label, batch_size):
        q = tf.FIFOQueue(capacity=2 * batch_size, dtypes=[tf.float32, tf.float32],
                         shapes=[(self.image_height, self.image_width, 3),
                                 (self.image_height, self.image_width, 1)])
        enqueue_op = q.enqueue([float_image, train_label])
        qr = tf.train.QueueRunner(q, [enqueue_op] * self.num_threads)
        tf.train.add_queue_runner(qr)
        batch_data, batch_label = q.dequeue_many(n=batch_size)
        return batch_data, batch_label


    def init_queue_segmentation(self, data_paths, labels):
        train_images = ops.convert_to_tensor(data_paths, dtype=dtypes.string)
        train_labels = ops.convert_to_tensor(labels, dtype=dtypes.string)
        # create input queues
        train_input_queue = tf.train.slice_input_producer(
            [train_images, train_labels],
            shuffle=True)
        return train_input_queue


    def process_batch_segmentation(self, train_input_queue):
        file_content_image = tf.read_file(train_input_queue[0])
        train_image = tf.image.decode_jpeg(file_content_image, channels=3)
        resized_image = tf.image.resize_images(train_image, [self.image_height, self.image_width])
        resized_image.set_shape([self.image_height, self.image_width, 3])
        float_image = tf.image.per_image_standardization(resized_image)

        file_content_mask = tf.read_file(train_input_queue[1])
        train_label = tf.image.decode_png(file_content_mask)
        train_label_resize = tf.image.resize_images(train_label, [self.image_height, self.image_width])
        float_image_gt = tf.reshape(tf.cast(train_label_resize,tf.float32),[self.image_height, self.image_width,1])
        float_image_gt.set_shape([self.image_height, self.image_width, 1])

        return float_image, float_image_gt