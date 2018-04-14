import os
import numpy as np
import tensorflow as tf
import  utils.TensorflowUtils as utils

VGG_MEAN = [103.939, 116.779, 123.68]# Mean value of pixels in R G and B channels

class Vgg16(object):

    def __init__(self, vgg_npy_path = None, num_classes=21):
        self.num_classes = num_classes
        if os.path.exists(vgg_npy_path):
            self.data_dict = np.load(vgg_npy_path, encoding='latin1').item()
            print("Pretrained model loaded from {}".format(vgg_npy_path))

        return

    def build(self, rgb, keep_prob):
        self.SumWeights = tf.constant(0.0, name="SumFiltersWeights") #Sum of weights of all filters for weight decay loss


        print("build model started")
        # rgb_scaled = rgb * 255.0

        # Convert RGB to BGR and substract pixels mean
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb)

        bgr = tf.concat(axis=3, values=[
            blue ,
            green,
            red ,
        ])
        bgr = tf.cast(bgr, tf.float32)
        # Layer 1
        self.conv1_1 = self.conv_layer(bgr, "conv1_1")  # Build Convolution layer and load weights
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")  # Build Convolution layer +Relu and load weights
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')  # Max Pooling
        # Layer 2
        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')
        # Layer 3
        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')
        # Layer 4
        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')
        # Layer 5
        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')
        ##-----------------------Build Net Fully connvolutional layers------------------------------------------------------------------------------------
        W6 = utils.weight_variable([7, 7, 512, 4096],
                                   name="W6")  # Create tf weight for the new layer with initial weights with normal random distrubution mean zero and std 0.02
        b6 = utils.bias_variable([4096], name="b6")  # Create tf biasefor the new layer with initial weights of 0
        self.conv6 = utils.conv2d_basic(self.pool5, W6,
                                        b6)  # Check the size of this net input is it same as input or is it 1X1
        self.relu6 = tf.nn.relu(self.conv6, name="relu6")
        # if FLAGS.debug: utils.add_activation_summary(relu6)
        self.relu_dropout6 = tf.nn.dropout(self.relu6,
                                           keep_prob=keep_prob)  # Apply dropout for traning need to be added only for training

        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")  # 1X1 Convloution
        b7 = utils.bias_variable([4096], name="b7")
        self.conv7 = utils.conv2d_basic(self.relu_dropout6, W7, b7)  # 1X1 Convloution
        self.relu7 = tf.nn.relu(self.conv7, name="relu7")
        # if FLAGS.debug: utils.add_activation_summary(relu7)
        self.relu_dropout7 = tf.nn.dropout(self.relu7,
                                           keep_prob=keep_prob)  # Another dropout need to be used only for training

        W8 = utils.weight_variable([1, 1, 4096, self.num_classes],
                                   name="W8")  # Basically the output num of classes imply the output is already the prediction this is flexible can be change however in multinet class number of 2 give good results
        b8 = utils.bias_variable([self.num_classes], name="b8")
        self.conv8 = utils.conv2d_basic(self.relu_dropout7, W8, b8)
        # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")
        # -------------------------------------Build Decoder --------------------------------------------------------------------------------------------------
        # now to upscale to actual image size
        deconv_shape1 = self.pool4.get_shape()  # Set the output shape for the the transpose convolution output take only the depth since the transpose convolution will have to have the same depth for output
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, self.num_classes],
                                     name="W_t1")  # Deconvolution/transpose in size 4X4 note that the output shape is of  depth NUM_OF_CLASSES this is not necessary in will need to be fixed if you only have 2 catagories
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        self.conv_t1 = utils.conv2d_transpose_strided(self.conv8, W_t1, b_t1, output_shape=tf.shape(
            self.pool4))  # Use strided convolution to double layer size (depth is the depth of pool4 for the later element wise addition
        self.fuse_1 = tf.add(self.conv_t1, self.pool4,
                             name="fuse_1")  # Add element wise the pool layer from the decoder

        deconv_shape2 = self.pool3.get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        self.conv_t2 = utils.conv2d_transpose_strided(self.fuse_1, W_t2, b_t2, output_shape=tf.shape(self.pool3))
        self.fuse_2 = tf.add(self.conv_t2, self.pool3, name="fuse_2")

        shape = tf.shape(rgb)
        W_t3 = utils.weight_variable([16, 16, self.num_classes, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([self.num_classes], name="b_t3")

        self.prob = utils.conv2d_transpose_strided(self.fuse_2, W_t3, b_t3,
                                                   output_shape=[shape[0], shape[1], shape[2], self.num_classes], stride=8)
        # --------------------Transform  probability vectors to label maps-----------------------------------------------------------------
        self.pred = tf.argmax(self.prob, dimension=3, name="Pred")

        print("FCN model built with vgg 16")
        return


    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = tf.cast(self.get_conv_filter(name), tf.float32)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def conv_layer_NoRelu(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)
            return bias

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc


    def get_conv_filter(self, name):
        var = tf.Variable(self.data_dict[name][0], name="filter_" + name)
        self.SumWeights += tf.nn.l2_loss(var)
        return var


    def get_bias(self, name):
        return tf.Variable(self.data_dict[name][1], name="biases_" + name)


    def get_fc_weight(self, name):
        return tf.Variable(self.data_dict[name][0], name="weights_" + name)








