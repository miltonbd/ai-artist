import tensorflow as tf


class Deconvnet(object):
    def __init__(self,image_height, image_width, channels, num_classes):
        self.image_height=image_height
        self.image_width=image_width
        self.channels=channels
        self.num_classes=num_classes

    def variable_with_weight_decay(self,name, shape, stddev, wd):
        dtype = tf.float32
        var = self.variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def variable_on_cpu(self, name, shape, initializer):
        with tf.device('/cpu:0'):
            dtype = tf.float32
            var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
        return var


    def build(self, input_batch):
        self.x=input_batch
        with tf.variable_scope('conv_1_1') as scope:
            conv_1_1 = self.conv_layer(self.x, [3, 3, 3, 64], 64, scope.name)

        with tf.variable_scope('conv_1_2') as scope:
            conv_1_2 = self.conv_layer(conv_1_1, [3, 3, 64, 64], 64, scope.name)
        with tf.device("/device:gpu:0"):
            pool_1, pool_1_argmax = self.pool_layer(conv_1_2)

        with tf.variable_scope('conv_2_1') as scope:

            conv_2_1 = self.conv_layer(pool_1, [3, 3, 64, 128], 128, 'conv_2_1')

        with tf.variable_scope('conv_2_2') as scope:

            conv_2_2 = self.conv_layer(conv_2_1, [3, 3, 128, 128], 128, 'conv_2_2')

        pool_2, pool_2_argmax = self.pool_layer(conv_2_2)

        with tf.variable_scope('conv_3_1') as scope:

            conv_3_1 = self.conv_layer(pool_2, [3, 3, 128, 256], 256, 'conv_3_1')
        with tf.variable_scope('conv_3_2') as scope:

            conv_3_2 = self.conv_layer(conv_3_1, [3, 3, 256, 256], 256, 'conv_3_2')

        with tf.variable_scope('conv_3_3') as scope:

            conv_3_3 = self.conv_layer(conv_3_2, [3, 3, 256, 256], 256, 'conv_3_3')

        pool_3, pool_3_argmax = self.pool_layer(conv_3_3)

        with tf.variable_scope('conv_4_1') as scope:

            conv_4_1 = self.conv_layer(pool_3, [3, 3, 256, 512], 512, 'conv_4_1')
        with tf.variable_scope('conv_4_2') as scope:

            conv_4_2 = self.conv_layer(conv_4_1, [3, 3, 512, 512], 512, 'conv_4_2')

        with tf.variable_scope('conv_4_3') as scope:

            conv_4_3 = self.conv_layer(conv_4_2, [3, 3, 512, 512], 512, 'conv_4_3')

        pool_4, pool_4_argmax = self.pool_layer(conv_4_3)

        with tf.variable_scope('conv_5_1') as scope:
            conv_5_1 = self.conv_layer(pool_4, [3, 3, 512, 512], 512, 'conv_5_1')

        with tf.variable_scope('conv_5_2') as scope:

            conv_5_2 = self.conv_layer(conv_5_1, [3, 3, 512, 512], 512, 'conv_5_2')
        with tf.variable_scope('conv_5_3') as scope:

            conv_5_3 = self.conv_layer(conv_5_2, [3, 3, 512, 512], 512, 'conv_5_3')

        pool_5, pool_5_argmax = self.pool_layer(conv_5_3)
        with tf.variable_scope('fc_6') as scope:

            fc_6 = self.conv_layer(pool_5, [7, 7, 512, 4096], 4096, 'fc_6')
        with tf.variable_scope('fc_7') as scope:

            fc_7 = self.conv_layer(fc_6, [1, 1, 4096, 4096], 4096, 'fc_7')
        with tf.variable_scope('fc6_deconv') as scope:

            deconv_fc_6 = self.deconv_layer(fc_7, [7, 7, 512, 4096], 512, 'fc6_deconv')

        unpool_5 = self.unpool_layer2x2(deconv_fc_6, pool_5_argmax, tf.shape(conv_5_3))
        with tf.variable_scope('deconv_5_3') as scope:

            deconv_5_3 = self.deconv_layer(unpool_5, [3, 3, 512, 512], 512, 'deconv_5_3')
        with tf.variable_scope('deconv_5_2') as scope:

            deconv_5_2 = self.deconv_layer(deconv_5_3, [3, 3, 512, 512], 512, 'deconv_5_2')
        with tf.variable_scope('deconv_5_1') as scope:

            deconv_5_1 = self.deconv_layer(deconv_5_2, [3, 3, 512, 512], 512, 'deconv_5_1')

        unpool_4 = self.unpool_layer2x2(deconv_5_1, pool_4_argmax, tf.shape(conv_4_3))

        with tf.variable_scope('deconv_4_3') as scope:
            deconv_4_3 = self.deconv_layer(unpool_4, [3, 3, 512, 512], 512, 'deconv_4_3')
        with tf.variable_scope('deconv_4_2') as scope:
            deconv_4_2 = self.deconv_layer(deconv_4_3, [3, 3, 512, 512], 512, 'deconv_4_2')
        with tf.variable_scope('deconv_4_1') as scope:
            deconv_4_1 = self.deconv_layer(deconv_4_2, [3, 3, 256, 512], 256, 'deconv_4_1')

        unpool_3 = self.unpool_layer2x2(deconv_4_1, pool_3_argmax, tf.shape(conv_3_3))
        with tf.variable_scope('deconv_3_3') as scope:
            deconv_3_3 = self.deconv_layer(unpool_3, [3, 3, 256, 256], 256, 'deconv_3_3')
        with tf.variable_scope('deconv_3_2') as scope:
            deconv_3_2 = self.deconv_layer(deconv_3_3, [3, 3, 256, 256], 256, 'deconv_3_2')
        with tf.variable_scope('deconv_3_1') as scope:
            deconv_3_1 = self.deconv_layer(deconv_3_2, [3, 3, 128, 256], 128, 'deconv_3_1')

        unpool_2 = self.unpool_layer2x2(deconv_3_1, pool_2_argmax, tf.shape(conv_2_2))
        with tf.variable_scope('deconv_2_2') as scope:

            deconv_2_2 = self.deconv_layer(unpool_2, [3, 3, 128, 128], 128, 'deconv_2_2')
        with tf.variable_scope('deconv_2_1') as scope:
            deconv_2_1 = self.deconv_layer(deconv_2_2, [3, 3, 64, 128], 64, 'deconv_2_1')

        unpool_1 = self.unpool_layer2x2(deconv_2_1, pool_1_argmax, tf.shape(conv_1_2))
        with tf.variable_scope('deconv_1_2') as scope:
            deconv_1_2 = self.deconv_layer(unpool_1, [3, 3, 64, 64], 64, 'deconv_1_2')
        with tf.variable_scope('deconv_1_1') as scope:
            deconv_1_1 = self.deconv_layer(deconv_1_2, [3, 3, 32, 64], 32, 'deconv_1_1')
        with tf.variable_scope('score_1') as scope:
            score_1 = self.deconv_layer(deconv_1_1, [1, 1, self.num_classes, 32], self.num_classes, 'score_1')

        self.logits = tf.reshape(score_1, (-1, self.num_classes))
        #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(expected, [-1]), logits=logits,
                                                                      # name='x_entropy')

        return score_1

    def conv_layer(self, x, W_shape, b_shape, scope, padding='SAME'):
        W = self.variable_with_weight_decay('weights', shape=W_shape, stddev=5e-2, wd=0.0)
        b =  self.variable_on_cpu('biases', b_shape, tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)
        pre_activation = tf.nn.bias_add(conv, b)
        #conv1_1 = tf.nn.relu(pre_activation, name=scope.name)
        return tf.nn.relu(pre_activation,name=scope)


    def deconv_layer(self, x, W_shape, b_shape, name, padding='SAME'):

        W = self.variable_with_weight_decay('weights', shape=W_shape, stddev=5e-2, wd=0.0)
        b = self.variable_on_cpu('biases', b_shape, tf.constant_initializer(0.1))
        x_shape = tf.shape(x)
        out_shape = tf.stack([x_shape[0], x_shape[1], x_shape[2], W_shape[2]])

        return tf.nn.conv2d_transpose(x, W, out_shape, [1, 1, 1, 1], padding=padding, name=name) + b

    def pool_layer(self, x):
        '''
        see description of build method
        '''
        with tf.device('/gpu:0'):
            return tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def deconv_layer(self, x, W_shape, b_shape, name, padding='SAME'):

        with tf.variable_scope(name) as scope:
            W = self.variable_with_weight_decay('weights', shape=W_shape, stddev=5e-2, wd=0.0)
            b = self.variable_on_cpu('biases', b_shape, tf.constant_initializer(0.1))

        x_shape = tf.shape(x)
        out_shape = tf.stack([x_shape[0], x_shape[1], x_shape[2], W_shape[2]])

        return tf.nn.conv2d_transpose(x, W, out_shape, [1, 1, 1, 1], padding=padding) + b

    def unravel_argmax(self, argmax, shape):
        output_list = []
        output_list.append(argmax // (shape[2] * shape[3]))
        output_list.append(argmax % (shape[2] * shape[3]) // shape[3])
        return tf.stack(output_list)

    def unravel_argmax(self, argmax, shape):
        output_list = []
        output_list.append(argmax // (shape[2] * shape[3]))
        output_list.append(argmax % (shape[2] * shape[3]) // shape[3])
        return tf.stack(output_list)

    def unpool_layer2x2(self, x, raveled_argmax, out_shape):
        argmax = self.unravel_argmax(raveled_argmax, tf.to_int64(out_shape))
        output = tf.zeros([out_shape[1], out_shape[2], out_shape[3]])

        height = tf.shape(output)[0]
        width = tf.shape(output)[1]
        channels = tf.shape(output)[2]

        t1 = tf.to_int64(tf.range(channels))
        t1 = tf.tile(t1, [((width + 1) // 2) * ((height + 1) // 2)])
        t1 = tf.reshape(t1, [-1, channels])
        t1 = tf.transpose(t1, perm=[1, 0])
        t1 = tf.reshape(t1, [channels, (height + 1) // 2, (width + 1) // 2, 1])

        t2 = tf.squeeze(argmax)
        t2 = tf.stack((t2[0], t2[1]), axis=0)
        t2 = tf.transpose(t2, perm=[3, 1, 2, 0])

        t = tf.concat([t2, t1], 3)
        indices = tf.reshape(t, [((height + 1) // 2) * ((width + 1) // 2) * channels, 3])

        x1 = tf.squeeze(x)
        x1 = tf.reshape(x1, [-1, channels])
        x1 = tf.transpose(x1, perm=[1, 0])
        values = tf.reshape(x1, [-1])

        delta = tf.SparseTensor(indices, values, tf.to_int64(tf.shape(output)))
        return tf.expand_dims(tf.sparse_tensor_to_dense(tf.sparse_reorder(delta)), 0)

    def unpool_layer2x2_batch(self, x, argmax):
        '''
        Args:
            x: 4D tensor of shape [batch_size x height x width x channels]
            argmax: A Tensor of type Targmax. 4-D. The flattened indices of the max
            values chosen for each output.
        Return:
            4D output tensor of shape [batch_size x 2*height x 2*width x channels]
        '''
        x_shape = tf.shape(x)
        out_shape = [x_shape[0], x_shape[1] * 2, x_shape[2] * 2, x_shape[3]]

        batch_size = out_shape[0]
        height = out_shape[1]
        width = out_shape[2]
        channels = out_shape[3]

        argmax_shape = tf.to_int64([batch_size, height, width, channels])
        argmax = unravel_argmax(argmax, argmax_shape)

        t1 = tf.to_int64(tf.range(channels))
        t1 = tf.tile(t1, [batch_size * (width // 2) * (height // 2)])
        t1 = tf.reshape(t1, [-1, channels])
        t1 = tf.transpose(t1, perm=[1, 0])
        t1 = tf.reshape(t1, [channels, batch_size, height // 2, width // 2, 1])
        t1 = tf.transpose(t1, perm=[1, 0, 2, 3, 4])

        t2 = tf.to_int64(tf.range(batch_size))
        t2 = tf.tile(t2, [channels * (width // 2) * (height // 2)])
        t2 = tf.reshape(t2, [-1, batch_size])
        t2 = tf.transpose(t2, perm=[1, 0])
        t2 = tf.reshape(t2, [batch_size, channels, height // 2, width // 2, 1])

        t3 = tf.transpose(argmax, perm=[1, 4, 2, 3, 0])

        t = tf.concat([t2, t3, t1], 4)
        indices = tf.reshape(t, [(height // 2) * (width // 2) * channels * batch_size, 4])

        x1 = tf.transpose(x, perm=[0, 3, 1, 2])
        values = tf.reshape(x1, [-1])

        delta = tf.SparseTensor(indices, values, tf.to_int64(out_shape))
        return tf.sparse_tensor_to_dense(tf.sparse_reorder(delta))