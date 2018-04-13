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

data_dir = "/home/milton/dataset/cifar/cifar10"
tran_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
num_gpus=2
image_height = 32
image_width = 32
num_channels = 3
num_classes=10
batch_size = 128 * num_gpus
num_threads=8 # keep 4 for 2 gpus

voc_class_labels = ['aeroplane','bicycle', 'bird', 'boat',
                    'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                    'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                    'train', 'tvmonitor']
cifar10_class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def cifar10_save():

    # batch_file = os.path.join(data_dir,"batches.meta")
    # with open(batch_file, mode='rb') as f:
    #     classes=_pickle.load(f, encoding="bytes")
    #     class_names=classes[b'label_names']
    #     for i in range(10):
    #         class_names[i] = class_names[i].decode('utf-8')
    #     print(class_names)
    for class_name in cifar10_class_labels:
        class_dir = os.path.join(tran_dir, class_name)
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)

    images=[]
    labels=[]
    for i in np.arange(1, 6):
        train_file = os.path.join(data_dir, 'data_batch_{}'.format(i))
        #print(train_file)
        with open(train_file, mode='rb') as f:
            data_dict = _pickle.load(f, encoding="bytes")
            labels_batch = data_dict[b'labels']
            data_batch = data_dict[b'data']
            for j in range(len(labels_batch)):
                images.append(data_batch[j])
                labels.append(labels_batch[j])

    for i in range(len(images)):
        img_flat = images[i]
        img_R = img_flat[0:1024].reshape((32, 32))
        img_G = img_flat[1024:2048].reshape((32, 32))
        img_B = img_flat[2048:3072].reshape((32, 32))
        img = np.dstack((img_R, img_G, img_B))
        #images_final = np.reshape(img, [32 * 32 * 3])
        images_final = img.astype(np.uint8)
        label_index = labels[i]
        label = cifar10_class_labels[label_index]
        file_path = os.path.join(tran_dir,label,"{}.jpg".format(i+1))
        imageio.imwrite(file_path, images_final)
        #print(label)

    test_images=[]
    test_labels=[]
    test_file = os.path.join(data_dir, 'test_batch')
    with open(test_file, mode='rb') as f:
        data_dict = _pickle.load(f, encoding="bytes")
        labels_batch = data_dict[b'labels']
        data_batch = data_dict[b'data']
        for j in range(len(labels_batch)):
            test_images.append(data_batch[j])
            test_labels.append(labels_batch[j])
    for i in range(len(test_images)):
        img_flat = test_images[i]
        img_R = img_flat[0:1024].reshape((32, 32))
        img_G = img_flat[1024:2048].reshape((32, 32))
        img_B = img_flat[2048:3072].reshape((32, 32))
        img = np.dstack((img_R, img_G, img_B))
        #images_final = np.reshape(img, [32 * 32 * 3])
        images_final = img.astype(np.uint8)
        label_index = labels[i]
        label = cifar10_class_labels[label_index]
        file_path = os.path.join(test_dir,label,"{}.jpg".format(i+1))
        imageio.imwrite(file_path, images_final)

#cifar10_save()

def get_test_files_cifar_10_classification():
   train_files=[]
   train_labels=[]
   for class_name in cifar10_class_labels:
       class_dir = os.path.join(test_dir, class_name)
       for file_name in os.listdir(class_dir):
           file_path = os.path.join(class_dir, file_name)
           train_files.append(file_path)
           train_labels.append(cifar10_class_labels.index(class_name))
   return train_files, train_labels


def get_train_files_cifar_10_classification():
   train_files=[]
   train_labels=[]
   for class_name in cifar10_class_labels:
       class_dir = os.path.join(tran_dir, class_name)
       for file_name in os.listdir(class_dir):
           file_path = os.path.join(class_dir, file_name)
           train_files.append(file_path)
           train_labels.append(cifar10_class_labels.index(class_name))
   return train_files, train_labels


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

# convert string into tensors
def process_batch(train_input_queue):
    file_content = tf.read_file(train_input_queue[0])
    train_image = tf.image.decode_jpeg(file_content, channels=3)
    resized_image = tf.image.resize_images(train_image, [image_height, image_width])
    resized_image.set_shape([image_height, image_width, 3])
    float_image = tf.image.per_image_standardization(resized_image)

    train_label_raw = train_input_queue[1]
    train_label_one_hot = tf.one_hot(train_label_raw, num_classes)
    train_label = tf.reshape(train_label_one_hot, [num_classes, ])
    return float_image, train_label

def make_queue(float_image, train_label):
    q = tf.FIFOQueue(capacity=5 * batch_size, dtypes=[tf.float32, tf.float32],
                     shapes=[(image_height, image_width, 3), (num_classes,)])
    enqueue_op = q.enqueue([float_image, train_label])
    qr = tf.train.QueueRunner(q, [enqueue_op] * num_threads)
    tf.train.add_queue_runner(qr)
    batch_data, batch_label = q.dequeue_many(n=batch_size)
    return batch_data, batch_label

def inti_queue(data_paths, labels):
    train_images = ops.convert_to_tensor(data_paths, dtype=dtypes.string)
    train_labels = ops.convert_to_tensor(labels, dtype=dtypes.int32)
    # create input queues
    train_input_queue = tf.train.slice_input_producer(
        [train_images, train_labels],
        shuffle=True)
    return  train_input_queue

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
        reshape = tf.layers.flatten(pool3)
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


    y_pred_cls = tf.argmax(softmax_linear, axis=1)

    return  softmax_linear, y_pred_cls



tf.reset_default_graph()

with  tf.device('/cpu:0'):
    # initialize the variables

    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True))
    sess.as_default()
    # initialize the queue threads to start to shovel data

    is_training = tf.placeholder(tf.bool, shape=None, name="is_training")

    #random.shuffle(all_filepath_labels)
    train_filepaths, all_train_labels = get_train_files_cifar_10_classification()
    float_image, train_label = process_batch(inti_queue(train_filepaths, all_train_labels))
    batch_data_train, batch_label_train = make_queue(float_image, train_label)

    test_filepaths, all_test_labels = get_train_files_cifar_10_classification()
    float_image, train_label = process_batch(inti_queue(test_filepaths, all_test_labels))
    batch_data_test, batch_label_test = make_queue(float_image, train_label)

    batch_data, batch_label = tf.cond(is_training,
                         lambda:(batch_data_train, batch_label_train),
                         lambda:(batch_data_test, batch_label_test))

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    optimizer = tf.train.GradientDescentOptimizer(1e-2)
    features_split = tf.split(batch_data, num_gpus, axis=0)
    labels_split = tf.split(batch_label, num_gpus, axis=0)

    tower_grads = []
    losses = []
    y_pred_classes = []

    for i in range(num_gpus):
        with tf.device('/gpu:{}'.format(i)):
            with tf.name_scope("tower_{}".format(i)) as scope:

                output, y_pred_class = core_model(features_split[i], labels_split[i])
                # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=labels_split[i]))
                # # losses = tf.get_collection('losses', scope)
                #
                # # Calculate the total loss for the current tower.
                # # loss = tf.add_n(losses, name='total_loss')
                tf.losses.softmax_cross_entropy(labels_split[i], output)
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
                    y_pred_classes.append(y_pred_class)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = average_gradients(tower_grads)
    apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

    losses_mean = tf.reduce_mean(losses)


    y_pred_classes_op=tf.reshape(tf.stack(y_pred_classes, axis=0),[-1])
    correct_prediction = tf.equal(y_pred_classes_op, tf.argmax(batch_label, axis=1))
    batch_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    sess.run(tf.global_variables_initializer())

    print("input pipeline ready")
    start = time.time()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord,sess=sess)

    print ("from the train set:")
    total_train_items = len(all_train_labels)
    batches_per_epoch = total_train_items//batch_size
    print("Total:{}, batch size: {}, batches per epoch: {}".format(total_train_items, batch_size, batches_per_epoch))
    try:
        for epoch in range(1):
            for step in range(batches_per_epoch):
                if coord.should_stop():
                    break

                _, loss_out = sess.run([apply_gradient_op,losses_mean],feed_dict={is_training:True})
                #print(logit_out[0])
                # We regularly check the loss
                if step % 10 == 0:
                    print('epoch:{}, step:{} - loss:{}'.format(epoch, step, loss_out))
                    #print("Accuracy: {}".format(sess.run(batch_accuracy, feed_dict={is_training:False})))
                #print(feed_batch_label[0])

    except:
        coord.request_stop()
    finally:
        coord.request_stop()
        coord.join(threads)
    coord.request_stop()
    coord.join(threads)
    sess.close()
    print("Time Taken {}".format(time.time()-start))