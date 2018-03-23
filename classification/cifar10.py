import tensorflow as tf
import numpy as np
import os
import _pickle

learning_rate = 1e-4
num_classes  = 10
gpu_nums = 2
data_dir = "/home/milton/dataset/cifar/cifar10" # meta, train, test


def load_cifar10_data():
    data=[]
    labels=[]

    for i in np.arange(1,6):
        train_file = os.path.join(data_dir,'data_batch_{}'.format(i))
        with open(train_file,  mode='rb') as f:
            data_dict=_pickle.load(f, encoding="bytes")
            labels_batch = data_dict[b'labels']
            data_batch = data_dict[b'data']
            for j in labels_batch:
                data.append(data_batch[j])
                labels.append(labels_batch[j])

    print("cifar10 loaded with {} items".format(len(labels)))
    return (np.asarray(data), np.asarray(labels))


def tower_loss(scope):
    with tf.get
    return


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
        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads




def main():

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        tower_grads = []
        for i in range(gpu_nums):
            with tf.device('/gpu:{}'.format(i)):
                with tf.name_scope("tower_".format(i)) as scope:

                    loss = tower_loss(scope)

                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()

                    grads = optimizer.apply_gradients(loss)

                    # Keep track of the gradients across all towers.
                    tower_grads.append(grads)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

        train_op = apply_gradient_op

        saver = tf.train.Saver(tf.all_variables())
        init = tf.initialize_all_variables()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True))
        sess.run(init)

        for itr in range(5):
            feed_dict = {
                images: images_data,
                labels: labels_data,
                keep_prob: 0.6
            }
            print("Iteration {})".format(itr))
            _, loss = sess.run([train_op, loss], feed_dict=feed_dict)

            # if itr % 50 == 0 and itr > 0:
            #     print("Saving Model to file in " + LOGS_DIR)
            #     saver.save(sess, LOGS_DIR + "model.ckpt", itr)  # Save model
            #
            # if itr % 10 == 0:
            #     # Calculate train loss
            #     feed_dict = {
            #         images: images_data,
            #         labels: labels_data,
            #         keep_prob: 1
            #     }
            #     TLoss = sess.run(loss, feed_dict=feed_dict)
            #     print("EPOCH=" + str(data_reader.epoch) + " Step " + str(itr) + "  Train Loss=" + str(TLoss))

    return




if __name__ == '__main__':
    load_cifar10_data()








