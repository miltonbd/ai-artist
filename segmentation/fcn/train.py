import  tensorflow as tf
import numpy as np
from segmentation.fcn.fcnvgg16 import FCNVGG16
from utils.data_reader import DataReader

np.random.seed(123)
tf.set_random_seed(123)

IMAGE_DIR = "/home/milton/dataset/segmentation/Materials_In_Vessels/Train_Images/"
LABEL_DIR = "/home/milton/dataset/segmentation/Materials_In_Vessels/LiquidSolidLabels/"
PRE_TRAIN_MODEL_PATH = "/home/milton/dataset/trained_models/vgg16.npy"

NUM_CLASSES = 4
EPOCHS = 5
BATCH_SIZE = 5
GPU_NUM = 2

tf.reset_default_graph()

sess = tf.Session()
#saver = tf.train.Saver()

data_reader = DataReader(image_dir=IMAGE_DIR, label_dir=LABEL_DIR, batch_size=BATCH_SIZE)
data_reader.loadDataSet()
ITERATIONS = EPOCHS * data_reader.total_train_count/BATCH_SIZE

print("Total Iterations {}".format(ITERATIONS))
net = FCNVGG16(PRE_TRAIN_MODEL_PATH,num_classes=NUM_CLASSES, dropout=0.5)

for itr in range(2):
    data_reader.nextBatch(itr)
#
#     feed_dict={}
#
#     sess.run(feed_dict=feed_dict)



print("Train finished")




