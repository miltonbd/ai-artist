import tensorflow as tf
import time
from classification.models.tensorflow import vgg16
from classification.cifar10.data_reader_cifar10 import *
from classification.queue_runner_utils_classification import QueueRunnerHelper

print("pid {}".format(os.getpid()))
data_dir = "/home/milton/dataset/cifar/cifar10"
tran_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
num_gpus = 1
image_height = 32
image_width = 32
num_channels = 3
num_classes = 10
batch_size_train_per_gpu = 250
batch_size_test_per_gpu = 250
batch_size_train = batch_size_train_per_gpu * num_gpus
batch_size_test = batch_size_test_per_gpu * num_gpus
num_threads = 2  # keep 4 for 2 gpus
learning_rate = 1e-3
epochs = 100
all_train_data = get_train_files_cifar_10_classification()
all_test_data = get_test_files_cifar_10_classification()
dropout = 0.5
#random.shuffle(all_filepath_labels)
train_filepaths, all_train_labels = get_train_files_cifar_10_classification()
test_filepaths, all_test_labels = get_test_files_cifar_10_classification()

total_train_items = len(all_train_labels)
total_test_items = len(test_filepaths)
batches_per_epoch_train = total_train_items // (num_gpus * batch_size_train)
batches_per_epoch_test = total_test_items // (num_gpus * batch_size_test)  # todo use multi gpu for testing.

print("Total Train:{}, batch size: {}, batches per epoch: {}".format(total_train_items, batch_size_train,
                                                                     batches_per_epoch_train))
print("Total Test:{}, batch size: {}, batches per epoch: {}".format(total_test_items, batch_size_test,
                                                                    batches_per_epoch_test))

is_training = tf.placeholder(tf.bool, shape=None, name="is_training")

queue_helper = QueueRunnerHelper(image_height,image_width,num_classes,num_threads)

train_float_image, train_label = queue_helper.process_batch(queue_helper.init_queue(train_filepaths, all_train_labels))
batch_data_train, batch_label_train = queue_helper.make_queue(train_float_image, train_label,batch_size_train)

test_float_image, test_label = queue_helper.process_batch(queue_helper.init_queue(test_filepaths, all_test_labels))
batch_data_test, batch_label_test = queue_helper.make_queue(test_float_image, test_label, batch_size_test)

batch_data, batch_label = tf.cond(is_training,
                     lambda:(batch_data_train, batch_label_train),
                     lambda:(batch_data_test, batch_label_test))
model= vgg16.Vgg16(num_classes=num_classes)
with tf.device('/gpu:1'):
    model.build(batch_data,0.5)
    logits=model.logits
    #print(logits.get_shape())
    #logits=tf.Print(logits,[logits.get_shape()])
    losses = tf.nn.sigmoid_cross_entropy_with_logits (None, tf.cast(batch_label, tf.float32), logits)

loss_op = tf.reduce_mean(losses)
y_pred_classes_op_batch = tf.nn.softmax(logits)
correct_prediction_batch = tf.cast(tf.equal(tf.argmax(y_pred_classes_op_batch,axis=1), tf.argmax(batch_label, axis=1)), tf.float32)
batch_accuracy = tf.reduce_mean(correct_prediction_batch)
#accuracy = tf.Print(accuracy, data=[accuracy], message="accuracy:")

# We add the training op ...
adam = tf.train.AdagradOptimizer(learning_rate)
train_op = adam.minimize(loss_op, name="train_op")

#
test_classes=[]
for test_index in range(batches_per_epoch_test):
    test_classes.append(correct_prediction_batch)

test_classes_op=tf.stack(test_classes, axis=0)
correct_prediction_test = tf.reshape(test_classes,[-1])
test_accuracy = tf.reduce_mean(correct_prediction_test)

print("input pipeline ready")
start = time.time()

# initialize the variables
global_step = tf.get_variable(
    'global_step', [],
    initializer=tf.constant_initializer(0), trainable=False)
sess=tf.Session()
# initialize the variables
sess.run(tf.global_variables_initializer())

print("input pipeline ready")
start = time.time()

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord,sess=sess)

try:
    for epoch in range(100):
        for step in range(batches_per_epoch_train):
            if coord.should_stop():
                break
            _, loss_out, batch_accuracy_out = sess.run([train_op,loss_op, batch_accuracy],feed_dict={is_training:True})

            if step % 50 == 0:
                print('epoch:{}, step:{} , loss:{}, batch accuracy:{}'.format(epoch, step, loss_out,batch_accuracy_out))

        #for test_index in range(batches_per_epoch_test):
            #test_classes.append(correct_prediction_batch)
        prediction_test_out, = sess.run([batch_accuracy],feed_dict={is_training: False})
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
print("Time Taken {}".format(time.time()-start))
