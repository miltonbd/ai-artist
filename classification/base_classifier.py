from abc import ABCMeta, abstractmethod
import tensorflow as tf
import time
from classification.queue_runner_utils_classification import QueueRunnerHelper
from utils.TensorflowUtils import average_gradients
from base.base_trainer import BaseTrainer


"""
This class is base class for all classifier. All new classifier must extend and implement this class.
"""

class BaseClassifier(BaseTrainer):
    __metaclass__ = ABCMeta

    def __init__(self, data_reader, model_params, model):
        BaseTrainer.__init__(self,data_reader,model_params,model)
        return

    """
    pass augmentation and various params here
    """
    def train(self):


        tf.reset_default_graph()
        tf.summary.FileWriterCache.clear()

        with  tf.device('/cpu:0'):
            # Assume that you have 12GB of GPU memory and want to allocate ~4GB:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
            sess = tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False,gpu_options=gpu_options))
            sess.as_default()

            global_step = tf.get_variable('global_step', [],initializer=tf.constant_initializer(0), trainable=False)

            # tf.summary.FileWriter('board_beginner',sess.graph)   # magic board
            writer = tf.summary.FileWriter(self.logdir)  # create writer

            is_training = tf.placeholder(tf.bool, shape=None, name="is_training")
            # random.shuffle(all_filepath_labels)
            all_train_filepaths, all_train_labels = self.data_reader.get_train_files()
            all_test_filepaths, all_test_labels = self.data_reader.get_validation_files()

            total_train_items = len(all_train_labels)
            total_test_items = len(all_test_labels)
            batches_per_epoch_train = total_train_items // (self.num_gpus * self.batch_size_train)
            batches_per_epoch_test = total_test_items // (self.num_gpus * self.batch_size_test)  # todo use multi gpu for testing.

            print("Total Train:{}, batch size: {}, batches per epoch: {}".format(total_train_items, self.batch_size_train,
                                                                          batches_per_epoch_train))
            print("Total Test:{}, batch size: {}, batches per epoch: {}".format(total_test_items, self.batch_size_test,
                                                                         batches_per_epoch_test))

            queue_helper = QueueRunnerHelper(self.image_height, self.image_width, self.num_classes, self.num_threads)
            train_float_image, train_label = queue_helper.process_batch(
                queue_helper.init_queue(all_train_filepaths, all_train_labels))

            # preprocess data

            # augment the trainng image here.

            batch_data_train, batch_label_train = queue_helper.make_queue(train_float_image, train_label, self.batch_size_train)

            test_float_image, test_label = queue_helper.process_batch(
                queue_helper.init_queue(all_test_filepaths, all_test_labels))
            batch_data_test, batch_label_test = queue_helper.make_queue(test_float_image, test_label, self.batch_size_test)

            batch_data, batch_label = tf.cond(is_training,
                                       lambda: (batch_data_train, batch_label_train),
                                       lambda: (batch_data_test, batch_label_test))

            optimizer = tf.train.AdagradOptimizer(self.learning_rate)

            features_split = tf.split(batch_data, self.num_gpus, axis=0)
            labels_split = tf.split(batch_label, self.num_gpus, axis=0)

            tower_grads = []
            tower_losses = []
            tower_y_pred_classes = []
            with tf.variable_scope(tf.get_variable_scope()):
                for i in range(self.num_gpus):
                    with tf.device('/gpu:{}'.format(i)):
                     with tf.name_scope("tower_{}".format(i)) as scope:
                         # logits, y_pred_class = core_model(features_split[i], labels_split[i])
                         x_input = tf.reshape(features_split[i], [-1, self.image_height, self.image_width, 3])
                         logits_per_gpu = self.model.build(features_split[i])

                         #logits_per_gpu = tf.reshape(model.conv8, [-1, self.num_classes])
                         # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=labels_split[i]))
                         # # losses = tf.get_collection('losses', scope)
                         #
                         # # Calculate the total loss for the current tower.
                         # # loss = tf.add_n(losses, name='total_loss')
                         # print(logits.get_shape())

                         tf.losses.softmax_cross_entropy(onehot_labels=labels_split[i], logits=logits_per_gpu)
                         tf.get_variable_scope().reuse_variables()
                         summaries_train = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                         update_ops = tf.get_collection(
                             tf.GraphKeys.UPDATE_OPS, scope)
                         updates_op = tf.group(*update_ops)
                         with tf.control_dependencies([updates_op]):
                             losses_per_gpu = tf.get_collection(tf.GraphKeys.LOSSES, scope)
                             total_loss_per_gpu = tf.add_n(losses_per_gpu, name='total_loss')
                             tower_losses.append(total_loss_per_gpu)
                             # Reuse variables for the next tower.

                         grads_per_gpu = optimizer.compute_gradients(total_loss_per_gpu)

                         # Keep track of the gradients across all towers.
                         tower_grads.append(grads_per_gpu)
                         soft_max_per_gpu = tf.nn.softmax(logits=logits_per_gpu)
                         predict_per_gpu = tf.argmax(soft_max_per_gpu, axis=1)
                         tower_y_pred_classes.append(predict_per_gpu)

            # We must calculate the mean of each gradient. Note that this is the
            # synchronization point across all towers.
            avg_grads = average_gradients(tower_grads)

            # Add histograms for gradients.
            for grad, var in avg_grads:
                if grad is not None:
                    summaries_train.append(tf.summary.histogram(var.op.name + '/gradients', grad))

            apply_gradient_op = optimizer.apply_gradients(avg_grads, global_step=global_step)

            # Add histograms for trainable variables.
            for var in tf.trainable_variables():
                summaries_train.append(tf.summary.histogram(var.op.name, var))

            batch_loss=tf.reshape(tf.stack(tower_losses, axis=0), [-1])
            loss_op = tf.reduce_mean(batch_loss)

            summaries_train.append(tf.summary.scalar("loss",loss_op))

            # model.build(batch_data, self.dropout)
            # logits = tf.reshape(model.conv8, [-1, self.num_classes])
            # # print(logits.get_shape())
            # # logits=tf.Print(logits,[logits.get_shape()])
            # losses = tf.nn.sigmoid_cross_entropy_with_logits(None, tf.cast(batch_label, tf.float32), logits)
            # loss_op = tf.reduce_mean(losses)

            y_pred_classes_op_batch = tf.reshape(tf.stack(tower_y_pred_classes, axis=0), [-1])
            labels_armax_batch =tf.argmax(batch_label, axis=1)
            correct_prediction_batch  = tf.equal(y_pred_classes_op_batch, labels_armax_batch )

            batch_accuracy = tf.reduce_mean(tf.cast(correct_prediction_batch, tf.float32))
            # accuracy = tf.Print(accuracy, data=[accuracy], message="accuracy:")
            summaries_train.append(tf.summary.scalar("batch_accuracy",batch_accuracy))

            # We add the training op ...
            train_op = apply_gradient_op

            test_classes = []
            for test_index in range(batches_per_epoch_test):
                test_classes.append(correct_prediction_batch)

            test_classes_op = tf.stack(test_classes, axis=0)
            correct_prediction_test = tf.reshape(test_classes_op, [-1])
            test_accuracy = tf.reduce_mean(tf.cast(correct_prediction_test, tf.float32))
            summaries_test=[]
            summaries_test.append(tf.summary.scalar("test_accuracy",test_accuracy))

            summary_op_train = tf.summary.merge(summaries_train)
            summary_op_test = tf.summary.merge(summaries_test)

            print("input pipeline ready")
            start = time.time()

            saver = tf.train.Saver()
            sess = tf.Session()
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

                     if step % 50 == 0:
                         print('epoch:{}, step:{} , loss:{}, batch accuracy:{}'.format(epoch, step, loss_out,
                                                                                       batch_accuracy_out))

                 # for test_index in range(batches_per_epoch_test):
                 # test_classes.append(correct_prediction_batch)
                 prediction_test_out, summary_out_test = sess.run([test_accuracy, summary_op_test], feed_dict={is_training: False})
                 writer.add_summary(summary_out_test, epoch)
                 print("Test Accuracy: {}".format(prediction_test_out))
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


