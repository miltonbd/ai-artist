import classification.models.tensorflow as tf
from utils.TensorflowUtils import *

"""
every model class will have a build() method
"""
class VGG16(object):

    def __init__(self,data_reader):
        self.data_reader = data_reader
        self.num_classes = data_reader.num_classes

    def build(self,input_batch):

        conv1_1 = self.conv_layer(input_batch, 'conv1_1', 64,  [5, 5], [1, 1] )
        pool1_1=self.max_pool_layer(self.normalization(conv1_1,'norm1_1'),'maxpool_1_1')

        conv1_2 = self.conv_layer(pool1_1,  'conv1_2', 64,[5,5],[1, 1])
        pool1_2 = self.max_pool_layer(self.normalization(conv1_2,'norm1_2'),'maxpool_1_2')


        conv2_1 = self.conv_layer(pool1_2, 'conv2_1',128, [3, 3], [1, 1])
        pool2_1 = self.max_pool_layer(self.normalization(conv2_1,'norm2_1'),'maxpool_2_1')
        flatten = tf.layers.flatten(pool2_1)
        logits = self.fully_connected_layer(flatten, 'output', self.num_classes)

        return  logits


    def fully_connected_layer(self, x, name, filters ):
        with tf.variable_scope(name) as scope:
            weights = variable_with_weight_decay('weights', shape=[x.get_shape()[-1], filters], stddev=0.04, wd=0.004)
            biases = variable_on_cpu('biases', [filters], tf.constant_initializer(0.1))
            fc = tf.nn.relu(tf.matmul(x, weights) + biases, name=scope.name)

        with tf.device('/cpu:0'):
            tf.summary.histogram('Fully connected layers/'+name, fc)
            tf.summary.scalar('Fully connected layers/'+name, tf.nn.zero_fraction(fc))
        return fc

    def conv_layer(self, x, name, filters_out, ksize=[3,3], stride=[1,1],):
        with tf.variable_scope(name) as scope:
            filters_in=x.get_shape()[-1]
            kernel_shape=[ksize[0], ksize[1], filters_in, filters_out]
            kernel = variable_with_weight_decay('weights', shape=kernel_shape, stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(x, kernel, [1, stride[0], stride[1], 1], padding='SAME')
            biases = variable_on_cpu('biases', [conv.get_shape()[3]], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv = tf.nn.relu(pre_activation, name=scope.name)
        with tf.device('/cpu:0'):
            tf.summary.histogram('Convolution_layers/'+name, conv)
            tf.summary.scalar('Convolution_layers/'+name, tf.nn.zero_fraction(conv))
        return conv

    def normalization(self,x, name):
        norm = tf.nn.lrn(x, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)
        return norm

    def max_pool_layer(self, x, name, k=[2,2],stride=[1,1], padding='SAME'):
        pool = tf.nn.max_pool(x, ksize=[1, k[0], k[1], 1], strides=[1, stride[0], stride[1], 1], padding=padding, name=name)
        return pool

    def dropout(self,x,keep_prob=0.6 ):
        return tf.nn.dropout(x,keep_prob)

