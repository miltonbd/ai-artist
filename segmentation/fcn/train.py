import  tensorflow as tf
import numpy as np
from segmentation.fcn.fcnvgg16 import FCNVGG16

np.random.seed(123)
tf.set_random_seed(123)

PRE_TRAIN_MODEL_PATH = "/home/milton/dataset/trained_models/vgg16.npy"
MAX_ITERATIONS = 2
BATCH_SIZE = 2
GPU_NUM = 2

tf.reset_default_graph()

sess = tf.Session()
#saver = tf.train.Saver()


net = FCNVGG16(PRE_TRAIN_MODEL_PATH)

# for itr in range(MAX_ITERATIONS):
#
#     feed_dict={}
#
#     sess.run(feed_dict=feed_dict)




