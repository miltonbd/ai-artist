import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from time import time
from classification.skin import data_loader
from classification.skin.models import simplenet

gpu_nums=2
TOWER_NAME="tower"

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
  #logits = cifar10.inference(images)

  # Build the portion of the Graph calculating the losses. Note that we will
  # assemble the total_loss using a custom function below.
  #_ = cifar10.loss(logits, labels)

  # Assemble all of the losses for the current tower only.
  losses = tf.get_collection('losses', scope)

  # Calculate the total loss for the current tower.
  total_loss = tf.add_n(losses, name='total_loss')

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    # loss_name = re.sub('%s_[0-9]*/' % cifar10.TOWER_NAME, '', l.op.name)
    # tf.summary.scalar(loss_name, l)

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

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from time import time
from classification.skin import data_loader
from classification.skin.models import simplenet

_IMG_SIZE = 224
_NUM_CHANNELS = 3
_BATCH_SIZE = 100
_CLASS_SIZE = 2
_ITERATION = 10000
_SAVE_PATH = "/home/milton/research/code-power/classification/skin/tensorboard/cifar-10/"


loader = data_loader.DataReaderISIC2017(_BATCH_SIZE,10,2)
loader.loadDataSet()

train_x, train_y, train_l = loader.getTrainDataForClassificationMelanoma()
#test_x, test_y, test_l = get_data_set("test")

x, y, output, global_step, y_pred_cls = simplenet.model()

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-3)

correct_prediction = tf.equal(y_pred_cls, tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar("Accuracy/train", accuracy)


merged = tf.summary.merge_all()
tower_grads = []
        losses =[]
        for i in range(gpu_nums):
            with tf.device('/gpu:{}'.format(i)):
                with tf.name_scope("tower_{}".format(i)) as scope:
                    start = i * _BATCH_SIZE
                    end = start + _BATCH_SIZE
                    loss = tower_loss(scope, x[start:end,:,:,:], y[start:end,:])
                    losses.append(loss)
                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()

                    grads = optimizer.compute_gradients(loss)

                    # Keep track of the gradients across all towers.
                    tower_grads.append(grads)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

saver = tf.train.Saver()
sess = tf.Session()
train_writer = tf.summary.FileWriter(_SAVE_PATH, sess.graph)


try:
    print("Trying to restore last checkpoint ...")
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=_SAVE_PATH)
    saver.restore(sess, save_path=last_chk_path)
    print("Restored checkpoint from:", last_chk_path)
except:
    print("Failed to restore checkpoint. Initializing variables instead.")
    sess.run(tf.global_variables_initializer())

def train(num_iterations):
    '''
        Train CNN
    '''
    for i in range(num_iterations):
        randidx = np.random.randint(len(train_x), size=_BATCH_SIZE)
        batch_xs = train_x[randidx]
        batch_ys = train_y[randidx]

        start_time = time()
        i_global, _ = sess.run([global_step, optimizer], feed_dict={x: batch_xs, y: batch_ys})
        duration = time() - start_time

        if (i_global % 10 == 0) or (i == num_iterations - 1):
            _loss, batch_acc = sess.run([loss, accuracy], feed_dict={x: batch_xs, y: batch_ys})
            msg = "Global Step: {0:>6}, accuracy: {1:>6.1%}, loss = {2:.2f} ({3:.1f} examples/sec, {4:.2f} sec/batch)"
            print(msg.format(i_global, batch_acc, _loss, _BATCH_SIZE / duration, duration))

        if (i_global % 500 == 0) or (i == num_iterations - 1):
            data_merged, global_1 = sess.run([merged, global_step], feed_dict={x: batch_xs, y: batch_ys})
            #acc = predict_test()

            # summary = tf.Summary(value=[
            #     tf.Summary.Value(tag="Accuracy/test", simple_value=acc),
            # ])
            # train_writer.add_summary(data_merged, global_1)
            # train_writer.add_summary(summary, global_1)

            saver.save(sess, save_path=_SAVE_PATH, global_step=global_step)
            print("Saved checkpoint.")


def predict_test(show_confusion_matrix=False):
    '''
        Make prediction for all images in test_x
    '''
    i = 0
    predicted_class = np.zeros(shape=len(test_x), dtype=np.int)
    while i < len(test_x):
        j = min(i + _BATCH_SIZE, len(test_x))
        batch_xs = test_x[i:j, :]
        batch_ys = test_y[i:j, :]
        predicted_class[i:j] = sess.run(y_pred_cls, feed_dict={x: batch_xs, y: batch_ys})
        i = j

    correct = (np.argmax(test_y, axis=1) == predicted_class)
    acc = correct.mean()*100
    correct_numbers = correct.sum()
    print("Accuracy on Test-Set: {0:.2f}% ({1} / {2})".format(acc, correct_numbers, len(test_x)))

    if show_confusion_matrix is True:
        cm = confusion_matrix(y_true=np.argmax(test_y, axis=1), y_pred=predicted_class)
        for i in range(_CLASS_SIZE):
            class_name = "({}) {}".format(i, test_l[i])
            print(cm[i, :], class_name)
        class_numbers = [" ({0})".format(i) for i in range(_CLASS_SIZE)]
        print("".join(class_numbers))

    return acc


if _ITERATION != 0:
    train(_ITERATION)


sess.close()
