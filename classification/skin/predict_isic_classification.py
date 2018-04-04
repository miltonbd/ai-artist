import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from classification.skin import data_loader
from classification.skin.models import simplenet

loader = data_loader.DataReaderISIC2017(128,10,2)
loader.loadDataSet()
test_x, test_y, test_l = loader.getTestDataForClassificationMelanoma()
x, y, output, global_step, y_pred_cls = simplenet.model()

_IMG_SIZE = 224
_NUM_CHANNELS = 3
_BATCH_SIZE = 64
_CLASS_SIZE = 2
_ITERATION = 10000

_SAVE_PATH = "/home/milton/research/code-power/classification/skin/tensorboard/cifar-10/"

saver = tf.train.Saver()
sess = tf.Session()

# http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/

try:
    print("Trying to restore last checkpoint ...")
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=_SAVE_PATH)
    saver.restore(sess, save_path=last_chk_path)
    print("Restored checkpoint from:", last_chk_path)
except:
    print("Failed to restore checkpoint. Initializing variables instead.")
    sess.run(tf.global_variables_initializer())

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

cm = confusion_matrix(y_true=np.argmax(test_y, axis=1), y_pred=predicted_class)
for i in range(_CLASS_SIZE):
    class_name = "({}) {}".format(i, test_l[i])
    print(cm[i, :], class_name)
class_numbers = [" ({0})".format(i) for i in range(_CLASS_SIZE)]
print("".join(class_numbers))


sess.close()
