import argparse
import sys
from classification.skin.models import simplenet
from classification.skin import data_loader
from classification.models.simplenetmultigpu import model as multiGpuModel

import numpy as np
from sklearn import metrics
from utils.data_reader_cifar10 import DataReaderCifar10
import utils.TensorflowUtils as tfutils  # place all utitilies funvtions here
from classification.models import vgg16
from utils.data_reader_cifar10 import *
from utils.queue_runner_utils import QueueRunnerHelper

# Train
step_num = 10001
gpu_nums = 2
batch_size = 200
batch_size_valid=200
batch_size_test=200
lr=0.001
image_height=32
image_width=32
image_channels=3
num_clasess=10
data_loader = DataReaderCifar10(batch_size, gpu_nums)
data_loader.loadDataSet()

from time import time

t0 = time()

import tensorflow as tf

tf.summary.FileWriterCache.clear()
# Import data
# construction phase

def model(input_images):
    x_images = tf.reshape(input_images,[-1,image_height,image_width,image_channels])
    with tf.name_scope("hidden"):
        W = tfutils._variable_with_weight_decay("weight",shape=[image_height*image_width*image_channels,num_clasess],stddev=0.04, wd=0.0)
        b = tfutils._variable_on_cpu('biases', [num_clasess],tf.zeros_initializer(10))
        logits = tf.matmul(input_images, W) + b
        with tf.device('/cpu:0'):
            tf.summary.histogram("weights", W)
            tf.summary.histogram("bias",b)
            y_historgram = tf.summary.histogram("activation",logits)
            tf.summary.image("images",x_images, 1)
        return  logits

def predict_valid(show_confusion_matrix=False):
    '''
        Make prediction for all images in test_x
    '''
    valid_x, valid_y, valid_l = loader.getValidationDataForClassificationMelanoma()
    i = 0
    y_pred = np.zeros(shape=len(valid_x), dtype=np.int)
    while i < len(valid_x):
        j = min(i + batch_size_valid, len(valid_x))
        batch_xs = valid_x[i:j, :]
        batch_ys = valid_y[i:j, :]
        y_pred[i:j] = sess.run(accuracy, feed_dict={images: batch_xs, labels: batch_ys,batch_size_gpu: batch_size_valid//2})
        i = j

    correct = (np.argmax(valid_y, axis=1) == y_pred)
    acc = correct.mean() * 100
    correct_numbers = correct.sum()
    print("Accuracy on Valid-Set: {0:.2f}% ({1} / {2})".format(acc, correct_numbers, len(valid_x)))

    y_true = np.argmax(valid_y, axis=1)
    cm = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
    for i in range(num_clasess):
        class_name = "({}) {}".format(i, valid_l[i])
        print(cm[i, :], class_name)
    class_numbers = [" ({0})".format(i) for i in range(num_clasess)]
    print("".join(class_numbers))

    auc = metrics.roc_auc_score(y_true, y_pred)
    print("Auc on Valid Set: {}".format(auc))

    f1_score = metrics.f1_score(y_true, y_pred)

    print("F1 score:  {}".format(f1_score))

    average_precision = metrics.average_precision_score(y_true, y_pred)

    print("average precsion on valid: {}".format(average_precision))

    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    return

tf.reset_default_graph()

with tf.Graph().as_default(), tf.device('/cpu:0'):
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

    images = tf.placeholder(tf.float32, shape=[None, image_height*image_width*image_channels],name="images")
    labels = tf.placeholder(tf.float32, shape=[None, num_clasess],name="labels")
    batch_size_gpu = tf.placeholder(tf.int32,name="batch_size_gpu")

    opt = tf.train.AdagradOptimizer(lr)
    tower_grads = []
    y_pred_classes = []
    losses = []

    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(gpu_nums):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % ("tower", i)) as scope:
                    # Dequeues one batch for the GPU
                    # Calculate the loss for one tower of the CIFAR model. This function
                    # constructs the entire CIFAR model but shares the variables across
                    # all towers.
                    start = i * batch_size_gpu
                    end = start + batch_size_gpu
                    logits=multiGpuModel(images[start:end,:],image_height,image_width,image_channels,num_clasess)
                    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels[start:end,:], logits=logits),
                                          name="softmax_cross_entropy")

                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()

                    # Retain the summaries from the final tower.
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                    # Calculate the gradients for the batch of data on this CIFAR tower.
                    grads_gpu = opt.compute_gradients(loss)

                    # Keep track of the gradients across all towers.
                    tower_grads.append(grads_gpu)
                    y_pred_classes.append(tf.argmax(tf.nn.softmax(logits),axis=1))
                    losses.append(loss)


    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = tfutils.average_gradients(tower_grads)

    #summaries.append(tf.summary.scalar('learning_rate', lr))

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        summaries.append(tf.summary.histogram(var.op.name, var))

    train_op=opt.apply_gradients(grads)

    losses_op = tf.reduce_mean(losses)

    # Test trained model
    y_pred_classes_op=tf.reshape(tf.stack(y_pred_classes, axis=0),[-1])

    correct_prediction = tf.equal(y_pred_classes_op, tf.argmax(labels, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy_scalar = tf.summary.scalar("accuracy", accuracy)


    # merge = tf.summary.merge_all()
    saver = tf.train.Saver()

    # execute phase
    sess = tf.InteractiveSession()
    # saver.restore(sess, "/tmp/model.ckpt")
    tf.global_variables_initializer().run()

    # tf.summary.FileWriter('board_beginner',sess.graph)   # magic board
    writer = tf.summary.FileWriter('logdir')  # create writer
    writer.add_graph(sess.graph)

    summary_op = tf.summary.merge(summaries)


    total_count = data_loader.total_train_count
    total_test_count = data_loader.total_test_count
    print("cifar 10 test")
    import math
    for step in range(step_num):
        batch_xs, batch_ys = data_loader.nextBatchTrain()
        startOffset = step%total_count
        endIndex = min(startOffset + batch_size*gpu_nums, total_count)
        #batch_xs = train_x[startOffset:endIndex, :]
        #batch_ys = train_y[startOffset:endIndex, :]
        #print("x,y shape : {}, {}".format(batch_xs.shape, batch_ys.shape))
        feed_train = {images: batch_xs, labels: batch_ys, batch_size_gpu:batch_size}
        _, merged_summary, accuracy_out,losses_op_out=sess.run([train_op,summary_op,accuracy,losses_op], feed_dict=feed_train)
        writer.add_summary(merged_summary)
        #print("loss: {}".format(losses_op_out))
        writer.add_summary(merged_summary, step)
        #
        # if step % 20 == 0:
        #     print("loss: {}".format(losses_op_out))
            # sum1 = sess.run(x_image, feed_dict=feed_train)
        #    sum2 = sess.run(accuracy_scalar, feed_dict=feed_train)
            #sum3 = sess.run(y_historgram, feed_dict=feed_train)
            #writer.add_summary(sum1, step)

            #writer.add_summary(sum3, step)

        if step % 200 == 0:
            y_pred = np.zeros(shape=[data_loader.total_test_count,], dtype=np.int32)
            test_iterations = total_test_count // (batch_size_test * gpu_nums)
            data_loader.itr_test=0
            #print("Test total:{}, batch test: {}, test iterations:{}".format(total_test_count, batch_size_test, test_iterations))
            for test_i in range(test_iterations-1):
                batch_xs, batch_ys = data_loader.nextTestBatch()
                #print(batch_xs[0])
                start = test_i * batch_size_valid
                end = min(start + batch_size_valid, total_test_count)
                #print("start: {}, end: {}".format(start,end))
                feed_test = {images: batch_xs, labels: batch_ys,
                             batch_size_gpu:batch_size_valid}
                correct_prediction_out=sess.run(correct_prediction, feed_dict=feed_test)
                y_pred[start:end] = correct_prediction_out[0:end-start]

            accuracy_test_set = np.mean(y_pred)
            print('step={}, accuracy={}, loss:{}'.format(step,accuracy_test_set, losses_op_out))

            #writer.add_summary()
            #predict_valid(100)
            #saver.save(sess, "./models-save/models")

