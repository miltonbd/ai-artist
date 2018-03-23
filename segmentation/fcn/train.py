import  tensorflow as tf
import numpy as np
from segmentation.fcn.fcnvgg16 import FCNVGG16
from utils.data_reader import DataReader
import time
import re

np.random.seed(123)
tf.set_random_seed(123)

IMAGE_DIR = "/home/milton/dataset/segmentation/Materials_In_Vessels/Train_Images/"
LABEL_DIR = "/home/milton/dataset/segmentation/Materials_In_Vessels/LiquidSolidLabels/"
PRE_TRAIN_MODEL_PATH = "/home/milton/dataset/trained_models/vgg16.npy"

NUM_CLASSES = 4
EPOCHS = 5
BATCH_SIZE = 5
GPU_NUM = 2
LEARNING_RATE = 1e-5
LOGS_DIR = "/home/milton/research/code-power-logs/fcnvgg16/"
TOWER_NAME = 'tower'
log_device_placement = True

# ..................... Create Data Reader ......................................#
data_reader = DataReader(image_dir=IMAGE_DIR, label_dir=LABEL_DIR, batch_size=BATCH_SIZE)
data_reader.loadDataSet()
ITERATIONS = EPOCHS * data_reader.total_train_count /(BATCH_SIZE * GPU_NUM)

print("Total Iterations {}".format(ITERATIONS))


def tower_loss(scope, images, labels, net,keep_prob):
  """Calculate the total loss on a single tower running the CIFAR model.
  Args:
    scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
    images: Images. 4D tensor of shape [batch_size, height, width, 3].
    labels: Labels. 1D tensor of shape [batch_size].
  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """

  # Build inference Graph.
  net.build(images, keep_prob)
  labels_squeez = tf.squeeze(labels, squeeze_dims=[3])
  logits = net.prob
  loss = tf.reduce_sum(
      (tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_squeez, logits=logits, name="loss")))

  # Build the portion of the Graph calculating the losses. Note that we will
  # assemble the total_loss using a custom function below.

  # Assemble all of the losses for the current tower only.
  tf.add_to_collection('losses', loss)
  losses = tf.get_collection('losses', scope)

  # Calculate the total loss for the current tower.
  total_loss = tf.add_n(losses, name='total_loss')

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    loss_name = re.sub('%s_[0-9]*/' % "tower", '', l.op.name)
    tf.summary.scalar(loss_name, l)

  return total_loss


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
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
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
    return average_grads



def train():
    tf.reset_default_graph()
    #................... placeholders for variables ...............................#
    images = tf.placeholder(tf.float32, shape= [None,None,None,3], name="input_image")
    labels = tf.placeholder(tf.int32, shape= [None, None, None, 1], name= "ground_truth")
    keep_prob = tf.placeholder(tf.float32, name= "dropout")


    #.................. Building net ..............................................#
    net = FCNVGG16(PRE_TRAIN_MODEL_PATH, num_classes=NUM_CLASSES)
    net.build(images, keep_prob )
    labels_squeez = tf.squeeze(labels, squeeze_dims=[3])
    logits = net.prob
    loss = tf.reduce_sum((tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_squeez,logits=logits, name="loss")))

    #.............. create solver for the net .....................................#
    trainable_vars = tf.trainable_variables()
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    grads = optimizer.compute_gradients(loss, trainable_vars)
    train_op = optimizer.apply_gradients(grads)


    #................. creating session ..............................................#
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(LOGS_DIR)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess,ckpt.model_checkpoint_path)
        print("Model restored.")
    start = time.time()

    for itr in range(5):
        images_data, labels_data = data_reader.nextBatch()
        #print(images_data.shape)
        #print(labels_data.shape)
        feed_dict = {
            images: images_data,
            labels: labels_data,
            keep_prob: 0.6
        }
        sess.run(train_op, feed_dict=feed_dict)

        if itr % 50 == 0 and itr > 0:
            print("Saving Model to file in " + LOGS_DIR)
            saver.save(sess, LOGS_DIR + "model.ckpt", itr)  # Save model

        if itr % 10==0:
            # Calculate train loss
            feed_dict = {
                images: images_data,
                labels: labels_data,
                keep_prob: 1
            }
            TLoss=sess.run(loss, feed_dict=feed_dict)
            print("EPOCH="+str(data_reader.epoch)+" Step "+str(itr)+ "  Train Loss="+str(TLoss))


    elapsed = time.time() - start
    print("Total elapsed {} ".format(elapsed))

def train_multi_gpu():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

    images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input_image")
    labels = tf.placeholder(tf.int32, shape=[None, None, None, 1], name="ground_truth")
    keep_prob = tf.placeholder(tf.float32, name="dropout")

    # .................. Building net ..............................................#
    net = FCNVGG16(PRE_TRAIN_MODEL_PATH, num_classes=NUM_CLASSES)
    trainable_vars = tf.trainable_variables()
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)

    # Calculate the gradients for each model tower.
    tower_grads = []
    with tf.variable_scope(tf.get_variable_scope()):
      for i in range(GPU_NUM):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % ("tower", i)) as scope:
            # Dequeues one batch for the GPU
            images_data, labels_data = data_reader.nextBatch()
            # Calculate the loss for one tower of the CIFAR model. This function
            # constructs the entire CIFAR model but shares the variables across
            # all towers.
            loss = tower_loss(scope, images_data, labels_data, net, keep_prob)

            # Reuse variables for the next tower.
            tf.get_variable_scope().reuse_variables()

            # Retain the summaries from the final tower.
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

            # Calculate the gradients for the batch of data on this CIFAR tower.
            grads = optimizer.compute_gradients(loss, trainable_vars)

            # Keep track of the gradients across all towers.
            tower_grads.append(grads)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = average_gradients(tower_grads)

    # Add a summary to track the learning rate.
    summaries.append(tf.summary.scalar('learning_rate', LEARNING_RATE))

    # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      summaries.append(tf.summary.histogram(var.op.name, var))


    # Group all updates to into a single train op.
    train_op = tf.group(apply_gradient_op)

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables())

    # Build the summary operation from the last tower summaries.
    summary_op = tf.summary.merge(summaries)

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    config = tf.ConfigProto(log_device_placement=log_device_placement, allow_soft_placement=True)
    sess = tf.Session(config=config)
    sess.run(init)

    summary_writer = tf.summary.FileWriter(LOGS_DIR, sess.graph)

    for itr in range(5):
        feed_dict = {
            images: images_data,
            labels: labels_data,
            keep_prob: 0.6
        }
        print("Iteration {})".format(itr))
        sess.run(train_op, feed_dict=feed_dict)

        if itr % 50 == 0 and itr > 0:
            print("Saving Model to file in " + LOGS_DIR)
            saver.save(sess, LOGS_DIR + "model.ckpt", itr)  # Save model

        if itr % 10 == 0:
            # Calculate train loss
            feed_dict = {
                images: images_data,
                labels: labels_data,
                keep_prob: 1
            }
            TLoss = sess.run(loss, feed_dict=feed_dict)
            print("EPOCH=" + str(data_reader.epoch) + " Step " + str(itr) + "  Train Loss=" + str(TLoss))

#train()
train_multi_gpu()
print("Train finished")




