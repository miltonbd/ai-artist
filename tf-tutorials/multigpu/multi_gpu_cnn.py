import tensorflow as tf
import os
import numpy as np
import math
from classification.skin import data_loader
import time
import random
from sklearn import metrics

_IMAGE_SIZE = 224
_IMAGE_CHANNELS = 3
_NUM_CLASSES = 2
_BATCH_SIZE = 100
_GPU_NUMS = 2
_EPOCHS = 10
learning_rate = 1e-4
gpu_nums=2

_SAVE_PATH = "/home/milton/research/code-power/MULTIGPU"
data_dir = "/home/milton/research/research-projects/tensorflow-multi-gpu/data"

def core_model(input_img, y,  num_classes):

    x_image = tf.reshape(input_img, [_BATCH_SIZE, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS], name='images')

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
        conv = tf.nn.conv2d(x_image, kernel, [1, 1, 1, 1], padding='SAME')
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

    with tf.variable_scope('fully_connected1') as scope:
        reshape = tf.layers.flatten(pool2)
        dim = reshape.get_shape()[1].value
        weights = variable_with_weight_decay('weights', shape=[dim, 192], stddev=0.04, wd=0.004)
        biases = variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    tf.summary.histogram('Fully connected layers/fc1', local3)
    tf.summary.scalar('Fully connected layers/fc1', tf.nn.zero_fraction(local3))

    with tf.variable_scope('output') as scope:
        weights = variable_with_weight_decay('weights', [192, _NUM_CLASSES], stddev=1 / 192.0, wd=0.0)
        biases = variable_on_cpu('biases', [_NUM_CLASSES], tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local3, weights), biases, name=scope.name)
    tf.summary.histogram('Fully connected layers/output', softmax_linear)

    y_pred_cls = tf.argmax(softmax_linear, axis=1)

    return  softmax_linear,  y_pred_cls

def variable_with_weight_decay(name, shape, stddev, wd):
        dtype = tf.float32
        var = variable_on_cpu( name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

def variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var

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

def predict_valid(show_confusion_matrix=False):
    '''
        Make prediction for all images in test_x
    '''
    valid_x, valid_y, valid_l = loader.getValidationDataForClassificationMelanoma()
    i = 0
    y_pred = np.zeros(shape=len(valid_x), dtype=np.int)
    output, y_pred_cls = core_model(images[start:end, :], labels[start:end, :], _NUM_CLASSES)

    while i < len(valid_x):
        j = min(i + _BATCH_SIZE , len(valid_x))
        batch_xs = valid_x[i:j, :]
        batch_ys = valid_y[i:j, :]
        y_pred[i:j] = sess.run(y_pred_cls, feed_dict={x: batch_xs, y: batch_ys})
        i = j

    correct = (np.argmax(valid_y, axis=1) == y_pred)
    acc = correct.mean() * 100
    correct_numbers = correct.sum()
    print("Accuracy on Valid-Set: {0:.2f}% ({1} / {2})".format(acc, correct_numbers, len(valid_x)))

    y_true = np.argmax(valid_y, axis=1)
    cm = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
    for i in range(_NUM_CLASSES):
        class_name = "({}) {}".format(i, valid_l[i])
        print(cm[i, :], class_name)
    class_numbers = [" ({0})".format(i) for i in range(_NUM_CLASSES)]
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
    loader = data_loader.DataReaderISIC2017(_BATCH_SIZE, _EPOCHS, gpu_nums)
    train_x, train_y, train_l = loader.getTrainDataForClassificationMelanoma()
    num_iterations = loader.iterations
    print("Iterations {}".format(num_iterations))
    total_count = loader.total_train_count
    step_local = int(math.ceil(_EPOCHS * total_count / _BATCH_SIZE))
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    images = tf.placeholder(tf.float32, shape=[_BATCH_SIZE * gpu_nums, 224*224*3])
    labels = tf.placeholder(tf.float32, shape=[_BATCH_SIZE * gpu_nums, 2])

    tower_grads = []
    losses = []
    y_pred_classes = []

    for i in range(gpu_nums):
        with tf.device('/gpu:{}'.format(i)):
            with tf.name_scope("tower_{}".format(i)) as scope:
                start = i * _BATCH_SIZE
                end = start + _BATCH_SIZE
                output, y_pred_class = core_model(images[start:end,:],labels[start:end,:], _NUM_CLASSES)
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=labels[start:end,:]))
                #losses = tf.get_collection('losses', scope)

                # Calculate the total loss for the current tower.
                #loss = tf.add_n(losses, name='total_loss')
                losses.append(loss)
                # Reuse variables for the next tower.
                tf.get_variable_scope().reuse_variables()

                grads = optimizer.compute_gradients(loss)
                # Keep track of the gradients across all towers.
                tower_grads.append(grads)
                y_pred_classes.append(y_pred_class)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = average_gradients(tower_grads)
    apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
    predict_out=tf.stack(y_pred_classes, axis=0)

    # variable_averages = tf.train.ExponentialMovingAverage(
    #     MOVING_AVERAGE_DECAY, global_step)
    # variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Group all updates to into a single train op.
    # train_op = tf.group(apply_gradient_op, variables_averages_op)
    train_op = apply_gradient_op

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.

    start = time.time()

    # saver = tf.train.Saver(tf.all_variables())

    saver = tf.train.Saver(tf.all_variables())
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False))
    train_writer = tf.summary.FileWriter(_SAVE_PATH, sess.graph)
    sess.run(init)
    step_global = sess.run(global_step)
    step_local = int(math.ceil(_EPOCHS * total_count / _BATCH_SIZE))
    epoch_done = int(math.ceil(step_global / (_BATCH_SIZE)))

    print("global:{}, local: {}, epochs done {}".format(step_global, step_local, epoch_done))
    if step_local < step_global:
        print("Training steps completed: global: {}, local: {}".format(step_global, step_local))
        exit()
    for epoch in range(epoch_done,_EPOCHS ):
        #print(total_count)
        shuffle_order=[i for i in range(total_count)]
        random.shuffle(shuffle_order)
        #print(shuffle_order)
        print("iterations {}".format(num_iterations))
        train_x = train_x[shuffle_order].reshape(total_count, -1)
        train_y = train_y[shuffle_order].reshape(total_count, -1)

        # this mehod is suitable when we load all training data in memory at once.
        for i in range(num_iterations):
            #print(num_iterations+_BATCH_SIZE)
            #print(loader.total_train_count)
            startIndex = _BATCH_SIZE * i * gpu_nums
            endIndex = min(startIndex + _BATCH_SIZE * gpu_nums, total_count )

            print("epoch:{}, iteration:{}, start:{}, end:{} ".format(epoch, i, startIndex, endIndex))

            batch_xs = train_x[startIndex:endIndex,:]
            batch_ys = train_y[startIndex:endIndex,:]
            print("feed data shape  {} , {}".format(batch_xs.shape, batch_ys.shape))
            #print(batch_ys)

            start_time = time.time()
            step_global_out, loss_out, y_pred_out = sess.run([global_step, losses, y_pred_classes], feed_dict={images: batch_xs, labels: batch_ys})
            steps =  + i

            if (step_global % 100 == 0) or (i == _EPOCHS * total_count - 1):
                print("epoch: {}, iteration: {}, loss: {}".format(epoch, i, loss_out))
                # _loss, batch_acc = sess.run([loss,], feed_dict={x: batch_xs, y: batch_ys})
                # duration = time.time() - start_time
                # msg = "Epoch: {0:}, Global Step: {1:>6}, accuracy: {2:>6.1%}, loss = {3:.2f} ({4:.1f} examples/sec, {5:.2f} sec/batch)"
                # print(msg.format(epoch,step_global, batch_acc, _loss, _BATCH_SIZE / duration, duration))

            #
            # if (step_global % 100 == 0) or (i == _EPOCHS * total_count  - 1):
            #     data_merged, global_1 = sess.run([merged, global_step], feed_dict={x: batch_xs, y: batch_ys})
            #     #acc = predict_test()
            #
            #     # summary = tf.Summary(value=[
            #     #     tf.Summary.Value(tag="Accuracy/test", simple_value=acc),
            #     # ])
            #     # train_writer.add_summary(data_merged, global_1)
            #     # train_writer.add_summary(summary, global_1)
            #
            #     saver.save(sess, save_path=_SAVE_PATH, global_step=global_step)
            #     print("Saved checkpoint.")
            #     predict_valid()





