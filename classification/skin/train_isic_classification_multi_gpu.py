import numpy as np
import tensorflow as tf
from sklearn import metrics
from time import time
from classification.skin import data_loader
from classification.skin.models import simplenet
import math

_IMG_SIZE = 224
_NUM_CHANNELS = 3
_BATCH_SIZE = 100
_CLASS_SIZE = 2
_EPOCHS = 500
_ITERATIONS = 100000
loader = data_loader.DataReaderISIC2017(_BATCH_SIZE,_EPOCHS,1)
#loader.loadDataSet()

#test_x, test_y, test_l = get_data_set("test")

x, y, output, global_step, y_pred_cls = simplenet.model()

_SAVE_PATH = "/home/milton/research/code-power/classification/skin/tensorboard/isic-2017-classification/"

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-3).minimize(loss, global_step=global_step)

correct_prediction = tf.equal(y_pred_cls, tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar("Accuracy/train", accuracy)

merged = tf.summary.merge_all()
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


def train():
    '''
        Train CNN
    '''
    train_x, train_y, train_l = loader.getTrainDataForClassificationMelanoma()

    num_iterations = loader.iterations
    print("Iterations {}".format(num_iterations))
    total_count = loader.total_train_count
    step_global = sess.run(global_step)
    step_local = int(math.ceil(_EPOCHS * total_count / _BATCH_SIZE))
    epoch_done = int(math.ceil(step_global/(_BATCH_SIZE )))

    print("global:{}, local: {}, epochs done {}".format(step_global, step_local, epoch_done))
    if step_local < step_global:
        print("Training steps completed: global: {}, local: {}".format(step_global, step_local))
        return
    for epoch in range(epoch_done,_EPOCHS ):
        shuffle_order = np.random.shuffle(np.arange(train_x.shape[0]))

        #print("iterations {}".format(num_iterations))
        train_x = train_x[shuffle_order].reshape(total_count, -1)
        train_y = train_y[shuffle_order].reshape(total_count, -1)

        for i in range(num_iterations):
            #print(num_iterations+_BATCH_SIZE)
            #print(loader.total_train_count)
            endIndex = min(num_iterations + _BATCH_SIZE, total_count )
            batch_xs = train_x[num_iterations:endIndex,:]
            batch_ys = train_y[num_iterations:endIndex,:]

            start_time = time()
            step_global, _ = sess.run([global_step, optimizer], feed_dict={x: batch_xs, y: batch_ys})
            duration = time() - start_time
            steps =  + i

            if (step_global % 10 == 0) or (i == _EPOCHS * total_count - 1):
                _loss, batch_acc = sess.run([loss, accuracy], feed_dict={x: batch_xs, y: batch_ys})
                msg = "Epoch: {0:}, Global Step: {1:>6}, accuracy: {2:>6.1%}, loss = {3:.2f} ({4:.1f} examples/sec, {5:.2f} sec/batch)"
                print(msg.format(epoch,step_global, batch_acc, _loss, _BATCH_SIZE / duration, duration))

            if (step_global % 100 == 0) or (i == _EPOCHS * total_count  - 1):
                data_merged, global_1 = sess.run([merged, global_step], feed_dict={x: batch_xs, y: batch_ys})
                #acc = predict_test()

                # summary = tf.Summary(value=[
                #     tf.Summary.Value(tag="Accuracy/test", simple_value=acc),
                # ])
                # train_writer.add_summary(data_merged, global_1)
                # train_writer.add_summary(summary, global_1)

                saver.save(sess, save_path=_SAVE_PATH, global_step=global_step)
                print("Saved checkpoint.")
                predict_valid()



def predict_valid(show_confusion_matrix=False):
    '''
        Make prediction for all images in test_x
    '''
    valid_x, valid_y, valid_l = loader.getValidationDataForClassificationMelanoma()
    i = 0
    y_pred = np.zeros(shape=len(valid_x), dtype=np.int)
    while i < len(valid_x):
        j = min(i + _BATCH_SIZE, len(valid_x))
        batch_xs = valid_x[i:j, :]
        batch_ys = valid_y[i:j, :]
        y_pred[i:j] = sess.run(y_pred_cls, feed_dict={x: batch_xs, y: batch_ys})
        i = j

    correct = (np.argmax(valid_y, axis=1) == y_pred)
    acc = correct.mean() * 100
    tf.summary.scalar("valid accuracy", acc)
    correct_numbers = correct.sum()
    print("Accuracy on Valid-Set: {0:.2f}% ({1} / {2})".format(acc, correct_numbers, len(valid_x)))

    y_true = np.argmax(valid_y, axis=1)
    cm = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
    for i in range(_CLASS_SIZE):
        class_name = "({}) {}".format(i, valid_l[i])
        print(cm[i, :], class_name)
    class_numbers = [" ({0})".format(i) for i in range(_CLASS_SIZE)]
    print("".join(class_numbers))

    auc = metrics.roc_auc_score(y_true, y_pred)
    tf.summary.scalar("valid auc", auc)
    print("Auc on Valid Set: {}".format(auc))

    f1_score = metrics.f1_score(y_true, y_pred)

    print("F1 score:  {}".format(f1_score))

    average_precision = metrics.average_precision_score(y_true, y_pred)

    print("average precsion on valid: {}".format(average_precision))

    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    return


if __name__ == '__main__':
    train()
    sess.close()



gpu_nums=2
TOWER_NAME="tower"

def tower_loss(scope, images, labels):
    net = MyCifar10Classifier(10)
    logits = net.inference(images)
    net.loss(logits,labels)
    losses = tf.get_collection('losses', scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')
    return total_loss

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
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
