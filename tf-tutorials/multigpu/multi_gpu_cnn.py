import tensorflow as tf
import os
import numpy as np


data_dir = "/home/milton/research/research-projects/tensorflow-multi-gpu/data"

num_gus = 2

def core_model(input_img, num_classes):
    net = tf.reshape(input_img, [-1,28,28,1])
    net = tf.layers.conv2d(inputs=net, filters=32, kernel_size=[5,5], padding='SAME' ,activation=tf.nn.relu())
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[2,2],strides=2)

    net = tf.layers.conv2d(inputs=net,filters=64, kernel_size=[5,5],activation=tf.nn.relu())
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[2,2], strides=[2,2])
    net = tf.layers.flatten(net)
    net = tf.layers.dense(inputs=net, units=1024,activation=tf.nn.relu())
    logits = tf.layers.dense(inputs=net, units=num_classes)
    return logits



tf.reset_default_graph()

with tf.variable_scope("a"):
    a = tf.placeholder(tf.float32, shape=[10,10],name)
    print(a.name)
