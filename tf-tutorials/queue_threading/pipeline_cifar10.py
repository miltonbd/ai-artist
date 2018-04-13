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
image_height = 32
image_width = 32
num_channels = 3
num_classes=10
batch_size = 128
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

z=model(batch_data,batch_label)

with tf.variable_scope('Loss'):
    losses = tf.nn.sigmoid_cross_entropy_with_logits(None, tf.cast(batch_label_train, tf.float32), z)
    loss_op = tf.reduce_mean(losses)

with tf.variable_scope('Accuracy'):
    y_pred = tf.cast(z > 0, tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred, batch_label_test), tf.float32))
    accuracy = tf.Print(accuracy, data=[accuracy], message="accuracy:")

# We add the training op ...
adam = tf.train.AdamOptimizer(1e-4)
train_op = adam.minimize(loss_op, name="train_op")

print("input pipeline ready")
start = time.time()
with tf.Session() as sess:
    # initialize the variables
    sess.run(tf.global_variables_initializer())

    # initialize the queue threads to start to shovel data
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    print ("from the train set:")
    for i in range(2000):
        _, loss, logit_out = sess.run([train_op, loss_op,y_pred],feed_dict={is_training:True})
        #print(logit_out[0])
        # We regularly check the loss
        if i % 50 == 0:
            print('iter:%d - loss:%f' % (i, loss))
            print("Accuracy: {}".format(sess.run(accuracy, feed_dict={is_training:False})))
        #print(feed_batch_label[0])
    coord.request_stop()
    coord.join(threads)
    sess.close()
    print("Time Taken {}".format(time.time()-start))