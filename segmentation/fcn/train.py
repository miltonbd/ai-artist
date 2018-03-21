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
BATCH_SIZE = 10
GPU_NUM = 2
LEARNING_RATE = 1e-5
LOGS_DIR = "/home/milton/research/code-power-logs/fcnvgg16/"

# ..................... Create Data Reader ......................................#
data_reader = DataReader(image_dir=IMAGE_DIR, label_dir=LABEL_DIR, batch_size=BATCH_SIZE)
data_reader.loadDataSet()
ITERATIONS = EPOCHS * data_reader.total_train_count / BATCH_SIZE

print("Total Iterations {}".format(ITERATIONS))


def tower_loss(scope, images, labels):
  """Calculate the total loss on a single tower running the CIFAR model.
  Args:
    scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
    images: Images. 4D tensor of shape [batch_size, height, width, 3].
    labels: Labels. 1D tensor of shape [batch_size].
  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """

  # Build inference Graph.
  logits = cifar10.inference(images)

  # Build the portion of the Graph calculating the losses. Note that we will
  # assemble the total_loss using a custom function below.
  _ = cifar10.loss(logits, labels)

  # Assemble all of the losses for the current tower only.
  losses = tf.get_collection('losses', scope)

  # Calculate the total loss for the current tower.
  total_loss = tf.add_n(losses, name='total_loss')

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    loss_name = re.sub('%s_[0-9]*/' % cifar10.TOWER_NAME, '', l.op.name)
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
    with tf.Graph().as_default(), tf.device('/cpu:0'):

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

        tower_grads = []

        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % ("Tower", i)) as scope:

        for itr in range(100):
            images_data, labels_data = data_reader.nextBatch(itr)
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

    # Calculate the learning rate schedule.
    num_batches_per_epoch = (cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                             FLAGS.batch_size)
    decay_steps = int(num_batches_per_epoch * cifar10.NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(cifar10.INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    cifar10.LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)

    # Create an optimizer that performs gradient descent.
    opt = tf.train.GradientDescentOptimizer(lr)

    # Get images and labels for CIFAR-10.
    images, labels = cifar10.distorted_inputs()
    batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
          [images, labels], capacity=2 * FLAGS.num_gpus)
    # Calculate the gradients for each model tower.
    tower_grads = []
    with tf.variable_scope(tf.get_variable_scope()):
      for i in xrange(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % (cifar10.TOWER_NAME, i)) as scope:
            # Dequeues one batch for the GPU
            image_batch, label_batch = batch_queue.dequeue()
            # Calculate the loss for one tower of the CIFAR model. This function
            # constructs the entire CIFAR model but shares the variables across
            # all towers.
            loss = tower_loss(scope, image_batch, label_batch)

            # Reuse variables for the next tower.
            tf.get_variable_scope().reuse_variables()

            # Retain the summaries from the final tower.
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

            # Calculate the gradients for the batch of data on this CIFAR tower.
            grads = opt.compute_gradients(loss)

            # Keep track of the gradients across all towers.
            tower_grads.append(grads)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = average_gradients(tower_grads)

    # Add a summary to track the learning rate.
    summaries.append(tf.summary.scalar('learning_rate', lr))

    # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      summaries.append(tf.summary.histogram(var.op.name, var))

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Group all updates to into a single train op.
    train_op = tf.group(apply_gradient_op, variables_averages_op)

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables())

    # Build the summary operation from the last tower summaries.
    summary_op = tf.summary.merge(summaries)

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = duration / FLAGS.num_gpus

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

      if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)



print("Train finished")




