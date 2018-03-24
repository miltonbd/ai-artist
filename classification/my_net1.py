import tensorflow as tf
import os
from classification.base_classifier import BaseClassifier


class MyNet1(BaseClassifier):
    def __int__(self):
        return

    def __init__(self, vgg_npy_path = None, num_classes=21):
        self.num_classes = num_classes
        if os.path.exists(vgg_npy_path):
            self.data_dict = np.load(vgg_npy_path, encoding='latin1').item()
            print("Pretrained model loaded from {}".format(vgg_npy_path))

        return

    def inference(self, images):
        with tf.variable_scope("conv1"):
            conv1 = self.conv_layer(X, [3, 3, 3, 4], [img_s, img_s, 4])
        with tf.variable_scope("conv2"):
            conv2 = self.conv_layer(conv1, [3, 3, 4, 4], [img_s, img_s, 4])
            pool1, img_s = self.max_pool_layer(conv2, img_s)

        cin = 4  # Channel In
        cout = 28  # Channel Out
        with tf.variable_scope("conv3"):
            conv3 = self.conv_layer(pool1, [3, 3, cin, cout], [img_s, img_s, cout])
        with tf.variable_scope("conv4"):
            conv4 = self.conv_layer(conv3, [3, 3, cout, cout], [img_s, img_s, cout])
            pool2, img_s = self.max_pool_layer(conv4, img_s)

        cin = cout  # 64
        cout = 56
        with tf.variable_scope("conv5"):
            conv5 = self.conv_layer(pool2, [3, 3, cin, cout], [img_s, img_s, cout])
        with tf.variable_scope("conv6"):
            conv6 = self.conv_layer(conv5, [3, 3, cout, cout], [img_s, img_s, cout])
        with tf.variable_scope("conv7"):
            conv7 = self.conv_layer(conv6, [3, 3, cout, cout], [img_s, img_s, cout])
        with tf.variable_scope("conv8"):
            conv8 = self.conv_layer(conv7, [3, 3, cout, cout], [img_s, img_s, cout])
            pool3, img_s = self.max_pool_layer(conv8, img_s)

        cin = cout  # 256
        cout = 12
        with tf.variable_scope("conv9"):
            conv9 = self.conv_layer(pool3, [3, 3, cin, cout], [img_s, img_s, cout])
        with tf.variable_scope("conv10"):
            conv10 = self.conv_layer(conv9, [3, 3, cout, cout], [img_s, img_s, cout])
        with tf.variable_scope("conv11"):
            conv11 = self.conv_layer(conv10, [3, 3, cout, cout], [img_s, img_s, cout])
        with tf.variable_scope("conv12"):
            conv12 = self.conv_layer(conv11, [3, 3, cout, cout], [img_s, img_s, cout])
            pool4, img_s = self.max_pool_layer(conv12, img_s)

        cin = cout  # 512
        cout = 12
        with tf.variable_scope("conv13"):
            conv13 = self.conv_layer(pool4, [3, 3, cin, cout], [img_s, img_s, cout])
        with tf.variable_scope("conv14"):
            conv14 = self.conv_layer(conv13, [3, 3, cout, cout], [img_s, img_s, cout])
        with tf.variable_scope("conv15"):
            conv15 = self.conv_layer(conv14, [3, 3, cout, cout], [img_s, img_s, cout])
        with tf.variable_scope("conv16"):
            conv16 = self.conv_layer(conv15, [3, 3, cout, cout], [img_s, img_s, cout])
            pool5, img_s = self.max_pool_layer(conv16, img_s)

        with tf.variable_scope("fc1"):
            n_in = img_s * img_s * cout  # 7*7*512
            n_out = 96  # 4096
            pool5_1d = tf.reshape(pool5, [batch_size, n_in])
            fc1 = self.fc_layer(pool5_1d, batch_size, n_in, n_out)
            fc1_drop = tf.nn.dropout(fc1, 0.6)
        with tf.variable_scope("fc2"):
            n_in = n_out  # 4096
            n_out = 96
            fc2 = self.fc_layer(fc1_drop, batch_size, n_in, n_out)
            fc2_drop = tf.nn.dropout(fc2, 0.6)
        with tf.variable_scope("fc3"):
            n_in = n_out
            y_ = self.fc_layer(fc2_drop, batch_size, n_in, n_labels, activation_fn=None)

        with tf.variable_scope('weights_norm'):
            weights_norm = tf.reduce_sum(
                input_tensor=weight_decay * tf.stack(
                    [tf.nn.l2_loss(i) for i in tf.get_collection('all_weights')]
                ),
                name='weights_norm'
            )
        tf.add_to_collection('losses', weights_norm)

        with tf.variable_scope('cross_entropy'):
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=y,
                                                        logits=y_))
        tf.add_to_collection('losses', cross_entropy)

        total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

        return total_loss

    def conv_layer(tensor, kernel_shape, bias_shape):
        """ Convolutional-ReLU layer """
        weights = tf.get_variable("weights", kernel_shape,
                                  initializer=tf.random_normal_initializer())
        tf.add_to_collection(weights, "all_weights")
        biases = tf.get_variable("biases", bias_shape,
                                 initializer=tf.constant_initializer(0.0))
        output = tf.nn.conv2d(tensor, weights, strides=[1, 1, 1, 1],
                              padding='SAME')
        return tf.nn.relu(output + biases)

    # pylint: disable=too-many-arguments, redefined-outer-name
    def fc_layer(vector, batch_size, n_in, n_out, activation_fn=tf.nn.relu):
        """ Fully Connected Layer"""
        weights = tf.get_variable("weights", [n_in, n_out],
                                  initializer=tf.random_normal_initializer())
        biases = tf.get_variable("biases", [batch_size, n_out],
                                 initializer=tf.constant_initializer(0.0))
        output = tf.add(tf.matmul(vector, weights), biases)
        if activation_fn is not None:
            output = activation_fn(output)
        return output

    def max_pool_layer(tensor, image_s):
        """ 2x2 Max pooling with stride 2"""
        image_s = int(image_s / 2)
        tensor = tf.nn.max_pool(tensor, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
        return tensor, image_s


if __name__ == '__main__':
    net = MyNet1()
    net.build()
