import numpy as np 
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data 
from Linear_Regression import trX

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=.01))

def model(X,W):
    return tf.matmul(X,W)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

X=tf.placeholder(dtype="float", name="X")
Y=tf.placeholder(dtype="float",name="Y")
W=init_weights([784,10])
y_pred=model(X, W)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_pred,Y))
train_op=tf.train.GradientDescentOptimizer(0.1).minimize(cost)
predict_op=tf.argmax(y_pred,1)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement=True
with tf.Session(config=config) as sess:
    tf.initialize_all_variables().run()
    for i in range(100):
        for start,end in zip(range(0,len(trX),256),range(256,len(trX)+1,256)):
            sess.run(train_op,feed_dict={X:trX[start:end],Y:trY[start:end]})
            test_result=(np.argmax(teY, axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX}))
                         
            print(i, np.mean(test_result) )
    