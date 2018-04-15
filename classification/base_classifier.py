from abc import ABCMeta, abstractmethod
import tensorflow as tf
import tensorflow as tf
import time
import _pickle
from classification.models import vgg16
from utils.data_reader_cifar10 import *
from utils.queue_runner_utils import QueueRunnerHelper
from sklearn import metrics


"""
This class is base class for all classifier. All new classifier must extend and implement this class.
"""

class BaseClassifier:
    __metaclass__ = ABCMeta

    def __init__(self, params, train_items=None, test_items=None, valid_items=None):
        self.data_dir = "/home/milton/dataset/cifar/cifar10" if not 'data_dir' in params else params['data_dir']
        self.tran_dir = os.path.join(self.data_dir, "train")
        self.test_dir = os.path.join(self.data_dir, "test")
        self.num_gpus = 2 if not 'num_gpus' in params else params['num_gpus']
        self.image_height = 32 if not 'image_height' in params else params['image_height']
        self.image_width = 32 if not 'image_width' in params else params['image_width']
        self.num_channels = 3
        self.num_classes = 10 if not 'num_classes' in params else params['num_classes']
        self.batch_size_train = 200 * self.num_gpus
        self.batch_size_test = 200 * self.num_gpus
        self.num_threads = 8  # keep 4 for 2 gpus
        self.learning_rate = 1e-3 if not 'learning_rate' in params else params['learning_rate']
        self.epochs = 100 if not 'epochs' in params else params['epochs']
        self.all_train_data = get_train_files_cifar_10_classification() if train_items == None else train_items
        self.all_test_data = get_test_files_cifar_10_classification() if test_items == None else test_items
        self.dropout=0.5  if not 'droput' in params else params['droput']

    def train(self):

     # random.shuffle(all_filepath_labels)
     all_train_filepaths, all_train_labels = self.all_train_data
     all_test_filepaths, all_test_labels =  self.all_test_data

     total_train_items = len(all_train_labels)
     total_test_items = len(all_test_filepaths)
     batches_per_epoch_train = total_train_items // (self.num_gpus * self.batch_size_train)
     batches_per_epoch_test = total_test_items // (self.num_gpus * self.batch_size_test)  # todo use multi gpu for testing.

     print("Total Train:{}, batch size: {}, batches per epoch: {}".format(total_train_items, self.batch_size_train,
                                                                          batches_per_epoch_train))
     print("Total Test:{}, batch size: {}, batches per epoch: {}".format(total_test_items, self.batch_size_test,
                                                                         batches_per_epoch_test))

     is_training = tf.placeholder(tf.bool, shape=None, name="is_training")

     queue_helper = QueueRunnerHelper(self.image_height, self.image_width, self.num_classes, self.num_threads)

     train_float_image, train_label = queue_helper.process_batch(
         queue_helper.init_queue(all_train_filepaths, all_train_labels))
     batch_data_train, batch_label_train = queue_helper.make_queue(train_float_image, train_label, self.batch_size_train)

     test_float_image, test_label = queue_helper.process_batch(
         queue_helper.init_queue(all_test_filepaths, all_test_labels))
     batch_data_test, batch_label_test = queue_helper.make_queue(test_float_image, test_label, self.batch_size_test)

     batch_data, batch_label = tf.cond(is_training,
                                       lambda: (batch_data_train, batch_label_train),
                                       lambda: (batch_data_test, batch_label_test))

     model = vgg16.Vgg16(num_classes=self.num_classes)
     model.build(batch_data, self.dropout)
     logits = tf.reshape(model.conv8, [-1, self.num_classes])
     # print(logits.get_shape())
     # logits=tf.Print(logits,[logits.get_shape()])
     losses = tf.nn.sigmoid_cross_entropy_with_logits(None, tf.cast(batch_label, tf.float32), logits)
     loss_op = tf.reduce_mean(losses)

     y_pred_classes_op_batch = tf.nn.softmax(logits)
     correct_prediction_batch = tf.cast(
         tf.equal(tf.argmax(y_pred_classes_op_batch, axis=1), tf.argmax(batch_label, axis=1)), tf.float32)
     batch_accuracy = tf.reduce_mean(correct_prediction_batch)
     # accuracy = tf.Print(accuracy, data=[accuracy], message="accuracy:")

     # We add the training op ...
     adam = tf.train.AdagradOptimizer(self.learning_rate)
     train_op = adam.minimize(loss_op, name="train_op")

     #
     test_classes = []
     for test_index in range(batches_per_epoch_test):
         test_classes.append(correct_prediction_batch)

     test_classes_op = tf.stack(test_classes, axis=0)
     correct_prediction_test = tf.reshape(test_classes, [-1])
     test_accuracy = tf.reduce_mean(correct_prediction_test)

     print("input pipeline ready")
     start = time.time()
     with  tf.device('/cpu:0'):
         # initialize the variables
         global_step = tf.get_variable(
             'global_step', [],
             initializer=tf.constant_initializer(0), trainable=False)
         sess = tf.Session()
         # initialize the variables
         sess.run(tf.global_variables_initializer())

         print("input pipeline ready")
         start = time.time()

         coord = tf.train.Coordinator()
         threads = tf.train.start_queue_runners(coord=coord, sess=sess)

         try:
             for epoch in range(self.epochs):
                 for step in range(batches_per_epoch_train):
                     if coord.should_stop():
                         break
                     _, loss_out, batch_accuracy_out = sess.run([train_op, loss_op, batch_accuracy],
                                                                feed_dict={is_training: True})

                     if step % 50 == 0:
                         print('epoch:{}, step:{} , loss:{}, batch accuracy:{}'.format(epoch, step, loss_out,
                                                                                       batch_accuracy_out))

                 # for test_index in range(batches_per_epoch_test):
                 # test_classes.append(correct_prediction_batch)
                 prediction_test_out, = sess.run([batch_accuracy], feed_dict={is_training: False})
                 print("Test Accuracy: {}".format(prediction_test_out))

         except Exception as e:
             print(e)
             coord.request_stop()
         finally:
             coord.request_stop()
             coord.join(threads)
         coord.request_stop()
         coord.join(threads)
         sess.close()
         print("Time Taken {}".format(time.time() - start))

    @abstractmethod
    def  build(self):
        pass

    @classmethod
    def a(self):
        pass

    @staticmethod
    def b():
        pass


