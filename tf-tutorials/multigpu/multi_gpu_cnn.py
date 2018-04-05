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


def tower_loss():
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)



tf.reset_default_graph()

with tf.variable_scope("a"):
    a = tf.placeholder(tf.float32, shape=[10,10],name)
    print(a.name)
