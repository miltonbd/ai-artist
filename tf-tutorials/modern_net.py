import numpy as np 
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("MNIST_data",one_hot=True)

trX,trY,teX,teY=mnist.train.images,mnist.train.labels,mnist.test.images,mnist.test.labels

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=.01))

def model(X,w_h,w_h2,w_0,p_keep,p_hidden):
    X=tf.nn.dropout(X,p_keep)
    h=tf.nn.relu(tf.matmul(X,w_h) )
    h=tf.nn.dropout(h,p_hidden)
    h2=tf.nn.relu(tf.matmul(h,w_h2))
    h2 = tf.nn.dropout(h2, p_hidden)
    
    return tf.matmul(h2,w_o)


X = tf.placeholder("float")
Y = tf.placeholder("float")

w_h = init_weights([784, 625])
w_h2 = init_weights([625, 1025])
w_o = init_weights([1025, 10])

p_keep_input = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)
    
# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()

    for i in range(100):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_input: 0.8, p_keep_hidden: 0.5})
        print(i, np.mean(np.argmax(teY, axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX, Y: teY,
                                                         p_keep_input: 1.0,
                                                         p_keep_hidden: 1.0})))
    