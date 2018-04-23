from abc import ABCMeta, abstractmethod
import tensorflow as tf
import tensorflow as tf
import time
import _pickle
from classification.models import vgg16
from utils.data_reader_cardava import *
from utils.queue_runner_utils_segmentation import QueueRunnerHelper
from sklearn import metrics
from utils.TensorflowUtils import average_gradients



"""
This class is base class for all classifier. All new classifier must extend and implement this class.
"""

class BaseSegmentation:
    __metaclass__ = ABCMeta

    def __init__(self, data_reader, model_params, model):
        self.data_reader=data_reader
        self.model_params=model_params
        self.num_gpus = model_params.num_gpus
        self.image_height = data_reader.image_height
        self.image_width = data_reader.image_width
        self.num_channels = data_reader.num_channels
        self.num_classes = data_reader.num_classes
        self.batch_size_train_per_gpu = model_params.batch_size_train_per_gpu
        self.batch_size_test_per_gpu = model_params.batch_size_test_per_gpu
        self.batch_size_train = self.batch_size_train_per_gpu * self.num_gpus
        self.batch_size_test = self.batch_size_test_per_gpu * self.num_gpus
        self.num_threads = data_reader.num_threads
        self.learning_rate = model_params.learning_rate
        self.epochs = model_params.epochs
        self.dropout=model_params.dropout
        self.optimizer= model_params.optimizer
        self.model = model
        self.logdir = model.logdir
        self.savedir = os.path.join(self.logdir, "saved_models/")

        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)

        """
        Adam: Auto learning rate decay with momentum
        
        """
    def showParamsBeforeTraining(self):
        print("num classes {}".format(self.num_classes))
        print("num gpus {}".format(self.num_gpus))


    """
    pass augmentation and various params here
    """
    def train(self):
        tf.reset_default_graph()
        tf.summary.FileWriterCache.clear()


        with  tf.device('/cpu:0'):
            sess = tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False))
            sess.as_default()
            self.showParamsBeforeTraining()

            global_step = tf.get_variable('global_step', [],initializer=tf.constant_initializer(0), trainable=False)

            # tf.summary.FileWriter('board_beginner',sess.graph)   # magic board
            writer = tf.summary.FileWriter(self.logdir)  # create writer

            is_training = tf.placeholder(tf.bool, shape=None, name="is_training")
            # random.shuffle(all_filepath_labels)
            all_train_filepaths, all_train_labels = self.data_reader.get_train_files()
            all_test_filepaths, all_test_labels =  self.data_reader.get_validation_files()

            total_train_items = len(all_train_labels)
            total_test_items = len(all_test_labels)
            batches_per_epoch_train = total_train_items // (self.num_gpus * self.batch_size_train)
            batches_per_epoch_test = total_test_items // (self.num_gpus * self.batch_size_test)  # todo use multi gpu for testing.

            print("Total Train:{}, batch size: {}, batches per epoch: {}".format(total_train_items, self.batch_size_train,
                                                                          batches_per_epoch_train))
            print("Total Test:{}, batch size: {}, batches per epoch: {}".format(total_test_items, self.batch_size_test,
                                                                         batches_per_epoch_test))

            queue_helper = QueueRunnerHelper(self.image_height, self.image_width, self.num_classes, self.num_threads)
            train_float_image, train_label = queue_helper.process_batch_segmentation(
                queue_helper.init_queue_segmentation(all_train_filepaths, all_train_labels))

            # preprocess data

            # augment the trainng image here.

            batch_data_train, batch_label_train = queue_helper.make_queue_segmentation(train_float_image, train_label, self.batch_size_train)

            test_float_image, test_label = queue_helper.process_batch_segmentation(
                queue_helper.init_queue_segmentation(all_test_filepaths, all_test_labels))
            batch_data_valid, batch_label_valid = queue_helper.make_queue_segmentation(test_float_image, test_label, self.batch_size_test)

            batch_data, batch_label = tf.cond(is_training,
                                       lambda: (batch_data_train, batch_label_train),
                                       lambda: (batch_data_valid, batch_label_valid))

            optimizer = tf.train.AdagradOptimizer(self.learning_rate)
            with tf.device('/gpu:0'):
                pred_annotation, logits_per_gpu = self.model.inference(batch_data)

                summaries_train = tf.get_collection(tf.GraphKeys.SUMMARIES)
                logits_per_gpu_int=tf.cast(logits_per_gpu,tf.float32)
                #label_batch_Reshape = tf.reshape(batch_label, [-1, self.num_classes])
                labels_t=tf.cast(tf.squeeze(batch_label, squeeze_dims=[3]),tf.int32)
                loss = tf.reduce_mean((
                     tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_per_gpu_int,
                                                                    labels=labels_t,
                                                                    name="entropy")))
            # tf.summary.scalar("entropy", loss)
            #
            trainable_var = tf.trainable_variables()
            # # tf.get_variable_scope().reuse_variables()
            #
            avg_grads = optimizer.compute_gradients(loss, var_list=trainable_var)
            apply_gradient_op = optimizer.apply_gradients(avg_grads, global_step=global_step)

            # Add histograms for trainable variables.
            for var in tf.trainable_variables():
                summaries_train.append(tf.summary.histogram(var.op.name, var))

            batch_loss = loss
            loss_op = tf.reduce_mean(batch_loss)

            summaries_train.append(tf.summary.scalar("loss", loss_op))

            y_pred_classes_op_batch = pred_annotation
            batch_label_int = tf.cast(batch_label, tf.int64)
            correct_prediction_batch = tf.equal(y_pred_classes_op_batch, batch_label_int)

            batch_accuracy = tf.reduce_mean(tf.cast(correct_prediction_batch, tf.float32))
            # accuracy = tf.Print(accuracy, data=[accuracy], message="accuracy:")
            summaries_train.append(tf.summary.scalar("batch_accuracy", batch_accuracy))

            # We add the training op ...
            train_op = apply_gradient_op

            test_classes = []
            for test_index in range(batches_per_epoch_test):
                test_classes.append(correct_prediction_batch)

            test_classes_op = tf.stack(test_classes, axis=0)
            correct_prediction_test = tf.reshape(test_classes_op, [-1])
            test_accuracy = tf.reduce_mean(tf.cast(correct_prediction_test, tf.float32))
            summaries_test = []
            summaries_test.append(tf.summary.scalar("test_accuracy", test_accuracy))

            summary_op_train = tf.summary.merge(summaries_train)
            summary_op_test = tf.summary.merge(summaries_test)

            print("input pipeline ready")
            start = time.time()

            saver = tf.train.Saver()

            # initialize the variables
            # initialize the variables
            try:
                print("Trying to restore last checkpoint ...")
                last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=self.savedir)
                saver.restore(sess, save_path=last_chk_path)
            except:
                print("Failed to restore checkpoint. Initializing variables instead.")
                sess.run(tf.global_variables_initializer())

            start = time.time()

            writer.add_graph(sess.graph)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)

            try:
             global_step_gone = sess.run(global_step)
             epoch_gone=int(global_step_gone)//batches_per_epoch_train

             try:
                 print("Restored {} global step checkpoint from:".format(global_step_gone, last_chk_path))
             except NameError:
                 pass

             print("startng training from {} epoch with {} GPUs".format(epoch_gone, self.num_gpus))

             for epoch in range(epoch_gone,self.epochs):
                 for step in range(batches_per_epoch_train):
                     if coord.should_stop():
                         break

                     _,merged_summary,loss_out, batch_accuracy_out, global_step_out = sess.run([train_op,summary_op_train,loss_op,batch_accuracy, global_step],
                                                               feed_dict={is_training: True})

                     if step % 5 == 0:
                        writer.add_summary(merged_summary, global_step_out)

                     if step % 5 == 0:
                        print('epoch:{}, step:{} , loss:{}, batch accuracy:{}'.format(epoch, step, loss_out,
                                                                                      batch_accuracy_out))

                 # for test_index in range(batches_per_epoch_test):
                 # test_classes.append(correct_prediction_batch)
                 #prediction_test_out, summary_out_test = sess.run([test_accuracy, summary_op_test], feed_dict={is_training: False})
                 #writer.add_summary(summary_out_test, epoch)
                 #print("Test Accuracy: {}".format(prediction_test_out))
                 saver.save(sess, save_path=self.savedir, global_step=global_step)
                 print("Saved checkpoint.")

            except Exception as e:
                print(e)
                coord.request_stop()
            except KeyboardInterrupt:
                saver.save(sess, save_path=self.savedir, global_step=global_step)
                print("Saved checkpoint globsl step {}".format(sess.run(global_step)))
                coord.request_stop()

            finally:
                coord.request_stop()
                coord.join(threads)

            coord.request_stop()
            coord.join(threads)
            sess.close()
            print("Time Taken {}".format(time.time() - start))

    def test(self):
        pass

    @abstractmethod
    def  build(self):
        pass

    @classmethod
    def a(self):
        pass

    @staticmethod
    def b():
        pass


