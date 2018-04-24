import tensorflow as tf
from classification.cifar10.data_reader_cifar10 import *
from classification.my_net1 import MyCifar10Classifier

model1 = "models/gpu_1_model.ckpt"
model2 = "models/gpu_2_model.ckpt"
saved_model_dir = "models"
batch_size = 200


with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.

    images = tf.placeholder(tf.float32, shape=[batch_size, 32, 32, 3])
    labels = tf.placeholder(tf.int32, shape=[batch_size, ])

    # Build a Graph that computes the logits predictions from the
    # inference model.
    net = MyCifar10Classifier(10)
    logits = net.inference(images)
    prob = tf.nn.softmax(logits)
    top_k_op = tf.nn.in_top_k(logits, labels, 1)
    saver = tf.train.Saver()

    # Calculate predictions.
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(saved_model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            data_loader = DataReaderCifar10(batch_size, 1, 1)
            data_loader.loadDataSet()
            total_true = 0
            for itr in range(data_loader.iterations_test):
                images_batch, labels_batch = data_loader.nextTestBatch()

                predict = sess.run(prob,feed_dict={
                    images:images_batch,
                    labels:labels_batch
                })
                print("for iteration {}".format(itr))
                print(predict[0,8])
                print(np.argmax(predict, axis=1))


        else:
            print('No checkpoint file found')


