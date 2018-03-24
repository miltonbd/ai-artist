import tensorflow as tf
import numpy as np
import os
import _pickle
from utils.data_reader_cifar10 import *
import time
from classification.vgg16 import Vgg16
from classification.my_net1 import MyCifar10Classifier
import sys

MOVING_AVERAGE_DECAY = 0.9999
epochs = 3
learning_rate = 1e-4
batch_size = 250
num_classes  = 10
gpu_nums = 2
data_dir = "/home/milton/dataset/cifar/cifar10" # meta, train, test
pretrain_model_path = "/home/milton/dataset/trained_models/vgg16.npy"
saved_model_dir = "models"
saved_model_path = os.path.join(saved_model_dir,"gpu_{}_model.ckpt".format(gpu_nums))

def tower_loss(scope, images, labels):
    net = MyCifar10Classifier(10)
    logits = net.inference(images)
    net.loss(logits,labels)
    losses = tf.get_collection('losses', scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')
    return total_loss

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
        grad = tf.concat(axis=0, values=grads)
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
        data_loader = DataReaderCifar10( batch_size,epochs,gpu_nums)
        data_loader.loadDataSet()
        iterations = data_loader.iterations
        print("Total iterations needed {}".format(iterations))
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        images = tf.placeholder(tf.float32, shape=[batch_size * gpu_nums, 32, 32, 3])
        labels = tf.placeholder(tf.float32, shape=[batch_size * gpu_nums, ])

        tower_grads = []
        losses =[]
        for i in range(gpu_nums):
            with tf.device('/gpu:{}'.format(i)):
                with tf.name_scope("tower_{}".format(i)) as scope:
                    start = i * batch_size
                    end = start + batch_size
                    loss = tower_loss(scope, images[start:end,:,:,:], labels[start:end])
                    losses.append(loss)
                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()

                    grads = optimizer.compute_gradients(loss)

                    # Keep track of the gradients across all towers.
                    tower_grads.append(grads)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

        # variable_averages = tf.train.ExponentialMovingAverage(
        #     MOVING_AVERAGE_DECAY, global_step)
        # variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # Group all updates to into a single train op.
        #train_op = tf.group(apply_gradient_op, variables_averages_op)
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

        for itr in range(data_loader.iterations):
            images_data,labels_data = data_loader.nextBatch()
            #print(labels_data.shape)
            feed_dict = {
                images: images_data,
                labels: labels_data
            }
            _, loss_all = sess.run([train_op, losses], feed_dict=feed_dict)
            msg = "Iteration {}, loss {})".format(itr, loss_all)
            if itr%20==0:
                print(msg)
            #sys.stdout.write(msg + "\r")
            #sys.stdout.flush()

        saver.save(sess, saved_model_path, global_step=data_loader.itr)
        print("Model Saved In {}".format(saved_model_path))

            #
            # if itr % 50 == 0 and itr > 0:
            #     print("Saving Model to file in " + LOGS_DIR)
            #     saver.save(sess, LOGS_DIR + "model.ckpt", itr)  # Save model
            #
            # if itr % 10 == 0:
            #     # Calculate train loss
            #     feed_dict = {
            #         images: images_data,
            #         labels: labels_data,
            #         keep_prob: 1
            #     }
            #     TLoss = sess.run(loss, feed_dict=feed_dict)
            #     print("EPOCH=" + str(data_reader.epoch) + " Step " + str(itr) + "  Train Loss=" + str(TLoss))
        end = time.time()
        print("Time elapsed {}".format(end - start))
    return




if __name__ == '__main__':
   main()









