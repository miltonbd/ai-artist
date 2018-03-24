import imageio
import tensorflow as tf
import matplotlib
import numpy as np

from utils.data_reader_cifar10 import DataReaderCifar10
import matplotlib.pyplot as plt

data_reader = DataReaderCifar10(10,2,1)
data_reader.loadDataSet()
images_batch, labels_batch = data_reader.nextBatch()
img=images_batch[3]
print(img.dtype)
imageio.imwrite("a.jpg",img)

images_batch = np.reshape(images_batch,[-1,32,32,3])
images = tf.placeholder(tf.float32,shape=[None, 32,32,3])

filter1 = tf.truncated_normal(shape=[3,3,3,3])
conv1 = tf.nn.conv2d(images,filter1,[1,1,1,1],padding='SAME')
bias1 = tf.

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    feed_dict = {
        images : images_batch
    }
    conv1_result = sess.run(conv1,feed_dict=feed_dict)
    imageio.imwrite("conv1.jpg", np.reshape(conv1_result[0],[32,32,3]))


