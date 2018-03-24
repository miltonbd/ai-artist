import tensorflow as tf
import numpy as np
import os
import _pickle
from utils.data_reader_cifar10 import *
import time
from classification.vgg16 import Vgg16
from classification.my_net1 import MyNet1

epochs = 5
learning_rate = 1e-4
batch_size = 10
num_classes  = 10
gpu_nums = 2
data_dir = "/home/milton/dataset/cifar/cifar10" # meta, train, test
PRE_TRAIN_MODEL_PATH = "/home/milton/dataset/trained_models/vgg16.npy"


def tower_loss(scope, images, labels):
    net = MyNet1()
    net.inference(images)

    return


def average_gradients(tower_grads):
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
        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads




def main():

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # load_cifar10_data()
        data_loader = DataReaderCifar10(data_dir, batch_size)
        data_loader.loadDataSet()
        iterations = epochs * data_loader.total_train_count // (batch_size * gpu_nums)
        print("Total iterations needed {}".format(iterations))
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        images = tf.placeholder(tf.float32, shape=[-1, 32, 32, 3])
        labels = tf.placeholder(tf.float32, shape=[-1, 10])

        tower_grads = []
        for i in range(gpu_nums):
            with tf.device('/gpu:{}'.format(i)):
                with tf.name_scope("tower_".format(i)) as scope:

                    loss = tower_loss(scope, images, labels)
                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()

                    grads = optimizer.apply_gradients(loss)

                    # Keep track of the gradients across all towers.
                    tower_grads.append(grads)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

        train_op = apply_gradient_op

        saver = tf.train.Saver(tf.all_variables())
        init = tf.initialize_all_variables()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.

        start = time.time()

        # saver = tf.train.Saver(tf.all_variables())
        init = tf.initialize_all_variables()

        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True))
        sess.run(init)


    return




if __name__ == '__main__':
   main()









