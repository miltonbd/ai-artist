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
from classification.models import vgg16
from utils.data_reader_cifar10 import *
from utils.queue_runner_utils import QueueRunnerHelper


data_dir = "/home/milton/dataset/cifar/cifar10"
tran_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
num_gpus=2
image_height = 32
image_width = 32
num_channels = 3
num_classes=10
batch_size_train = 100 * num_gpus
batch_size_test = 100 * num_gpus
num_threads=8 # keep 4 for 2 gpus
epochs=100

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


def model(batch_data, batch_label):
    input=tf.reshape(batch_data,[-1, 32*32*3])
    with tf.variable_scope('FullyConnected'):
        w = tf.get_variable('w', shape=[3072, 1024], initializer=tf.random_normal_initializer(stddev=1e-1))
        b = tf.get_variable('b', shape=[1024], initializer=tf.constant_initializer(0.1))
        z = tf.matmul(input, w) + b
        y = tf.nn.relu(z)

        w2 = tf.get_variable('w2', shape=[1024, num_classes], initializer=tf.random_normal_initializer(stddev=1e-1))
        b2 = tf.get_variable('b2', shape=[num_classes], initializer=tf.constant_initializer(0.1))
        z = tf.matmul(y, w2) + b2
        tf.get_variable_scope().reuse_variables()

    return z


def core_model(input_img, y):

    #x_image =tf.reshape(batch_data,[-1, 32*32*3])

    def variable_with_weight_decay(name, shape, stddev, wd):
        dtype = tf.float32
        var = variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def variable_on_cpu(name, shape, initializer):
        with tf.device('/cpu:0'):
            dtype = tf.float32
            var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
        return var

    with tf.variable_scope('conv1') as scope:
        kernel = variable_with_weight_decay('weights', shape=[5, 5, 3, 64], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(input_img, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
    tf.summary.histogram('Convolution_layers/conv1', conv1)
    tf.summary.scalar('Convolution_layers/conv1', tf.nn.zero_fraction(conv1))

    norm1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    pool1 = tf.nn.max_pool(norm1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    with tf.variable_scope('conv2') as scope:
        kernel = variable_with_weight_decay('weights', shape=[5, 5, 64, 64], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
    tf.summary.histogram('Convolution_layers/conv2', conv2)
    tf.summary.scalar('Convolution_layers/conv2', tf.nn.zero_fraction(conv2))

    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    with tf.variable_scope('conv3') as scope:
        kernel = variable_with_weight_decay('weights', shape=[3, 3, 64, 128], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name=scope.name)
    tf.summary.histogram('Convolution_layers/conv3', conv3)
    tf.summary.scalar('Convolution_layers/conv3', tf.nn.zero_fraction(conv3))

    with tf.variable_scope('conv4') as scope:
        kernel = variable_with_weight_decay('weights', shape=[3, 3, 128, 128], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(pre_activation, name=scope.name)
    tf.summary.histogram('Convolution_layers/conv4', conv4)
    tf.summary.scalar('Convolution_layers/conv4', tf.nn.zero_fraction(conv4))

    with tf.variable_scope('conv5') as scope:
        kernel = variable_with_weight_decay('weights', shape=[3, 3, 128, 128], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(pre_activation, name=scope.name)
    tf.summary.histogram('Convolution_layers/conv5', conv5)
    tf.summary.scalar('Convolution_layers/conv5', tf.nn.zero_fraction(conv5))

    norm3 = tf.nn.lrn(conv5, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
    pool3 = tf.nn.max_pool(norm3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    with tf.variable_scope('fully_connected1') as scope:
        reshape = tf.layers.flatten(tf.nn.dropout(pool3,keep_prob=0.5))
        dim = reshape.get_shape()[1].value
        weights = variable_with_weight_decay('weights', shape=[dim, 384], stddev=0.04, wd=0.004)
        biases = variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    tf.summary.histogram('Fully connected layers/fc1', local3)
    tf.summary.scalar('Fully connected layers/fc1', tf.nn.zero_fraction(local3))

    with tf.variable_scope('fully_connected2') as scope:
        weights = variable_with_weight_decay('weights', shape=[384, 192], stddev=0.04, wd=0.004)
        biases = variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    tf.summary.histogram('Fully connected layers/fc2', local4)
    tf.summary.scalar('Fully connected layers/fc2', tf.nn.zero_fraction(local4))

    with tf.variable_scope('output') as scope:
        weights = variable_with_weight_decay('weights', [192, num_classes], stddev=1 / 192.0, wd=0.0)
        biases = variable_on_cpu('biases', [num_classes], tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    tf.summary.histogram('Fully connected layers/output', softmax_linear)


    y_pred_cls = tf.argmax(tf.nn.softmax(softmax_linear), axis=1)
    tf.get_variable_scope().reuse_variables()

    return  softmax_linear, y_pred_cls



tf.reset_default_graph()

with  tf.device('/cpu:0'):
    # initialize the variables
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False))
    sess.as_default()
    # initialize the queue threads to start to shovel data

    is_training = tf.placeholder(tf.bool, shape=None, name="is_training")

    # random.shuffle(all_filepath_labels)
    train_filepaths, all_train_labels = get_train_files_cifar_10_classification()

    queue_helper = QueueRunnerHelper(image_height, image_width, num_classes, num_threads)

    float_image, train_label = queue_helper.process_batch(queue_helper.init_queue(train_filepaths, all_train_labels))
    batch_data_train, batch_label_train = queue_helper.make_queue(float_image, train_label, batch_size_train)

    test_filepaths, all_test_labels = get_train_files_cifar_10_classification()
    float_image, train_label = queue_helper.process_batch(queue_helper.init_queue(test_filepaths, all_test_labels))
    batch_data_test, batch_label_test = queue_helper.make_queue(float_image, train_label, batch_size_test)

    batch_data, batch_label = tf.cond(is_training,
                         lambda:(batch_data_train, batch_label_train),
                         lambda:(batch_data_test, batch_label_test))

    model = vgg16.Vgg16(num_classes=num_classes)
    model.build(batch_data, 0.5)
    tf.get_variable_scope().reuse_variables()

    logits =  tf.reshape(model.conv8, [-1, num_classes])

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    optimizer = tf.train.GradientDescentOptimizer(1e-2)
    features_split = tf.split(batch_data, num_gpus, axis=0)
    labels_split = tf.split(batch_label, num_gpus, axis=0)

    tower_grads = []
    losses = []
    y_pred_classes = []
    vgg=vgg16.Vgg16(num_classes=10)

    for i in range(num_gpus):
        with tf.device('/gpu:{}'.format(i)):
            with tf.name_scope("tower_{}".format(i)) as scope:
                #logits, y_pred_class = core_model(features_split[i], labels_split[i])
                x_input=tf.reshape(features_split[i],[-1,32,32,3])
                vgg.build(x_input,0.5)
                # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=labels_split[i]))
                # # losses = tf.get_collection('losses', scope)
                #
                # # Calculate the total loss for the current tower.
                # # loss = tf.add_n(losses, name='total_loss')
                tf.losses.softmax_cross_entropy(labels_split[i], tf.reshape(logits,[100,10]))
                update_ops = tf.get_collection(
                    tf.GraphKeys.UPDATE_OPS, scope)
                updates_op = tf.group(*update_ops)
                with tf.control_dependencies([updates_op]):
                    losses = tf.get_collection(tf.GraphKeys.LOSSES, scope)
                    total_loss = tf.add_n(losses, name='total_loss')
                    losses.append(total_loss)
                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()

                    grads = optimizer.compute_gradients(total_loss)
                    # Keep track of the gradients across all towers.
                    tower_grads.append(grads)
                    soft_max = tf.nn.softmax(logits=logits)
                    predict = tf.argmax(soft_max, axis=1)
                    y_pred_classes.append(predict)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    avg_grads = average_gradients(tower_grads)
    apply_gradient_op = optimizer.apply_gradients(avg_grads, global_step=global_step)

    losses_mean = tf.reduce_mean(losses)


    y_pred_classes_op_batch=tf.reshape(tf.stack(y_pred_classes, axis=0), [-1])
    correct_prediction_batch = tf.cast(tf.equal(y_pred_classes_op_batch, tf.argmax(batch_label, axis=1)), tf.float32)
    batch_accuracy = tf.reduce_mean(correct_prediction_batch)

    print ("from the train set:")
    total_train_items = len(all_train_labels)
    total_test_items = len(test_filepaths)
    batches_per_epoch_train = total_train_items//(num_gpus*batch_size_train)
    batches_per_epoch_test = total_test_items//(batch_size_test) # todo use multi gpu for testing.

    print("Total Train:{}, batch size: {}, batches per epoch: {}".format(total_train_items, batch_size_train, batches_per_epoch_train))
    print("Total Test:{}, batch size: {}, batches per epoch: {}".format(total_test_items, batch_size_test,
                                                                        batches_per_epoch_test))
    sess.run(tf.global_variables_initializer())

    print("input pipeline ready")
    start = time.time()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord,sess=sess)

    test_classes=[]
    for test_index in range(batches_per_epoch_test):
        test_classes.append(correct_prediction_batch)

    test_classes_op=tf.stack(test_classes, axis=0)
    correct_prediction_test = tf.reshape(test_classes,[-1])
    test_accuracy = tf.reduce_mean(correct_prediction_test)

    try:
        for epoch in range(100):
            for step in range(batches_per_epoch_train):
                if coord.should_stop():
                    break
                _, loss_out, batch_accuracy_out = sess.run([apply_gradient_op,losses_mean, batch_accuracy],feed_dict={is_training:True})
                #print(logit_out[0])
                # We regularly check the loss
                # if step % 50 == 0:
                #     print('epoch:{}, step:{} , loss:{}, batch accuracy:{}'.format(epoch, step, loss_out,batch_accuracy_out))
                #print(feed_batch_label[0])


            prediction_test_out, = sess.run([test_accuracy],feed_dict={is_training: False})
            print("Test Accuracy: {}".format(prediction_test_out))

    except Exception as e:
        print(e)
        coord.request_stop()
    finally:
        coord.request_stop()
        coord.join(threads)
    coord.request_stop()
    coord.join(threads)
    sess.close()
    print("Time Taken {}".format(time.time()-start))