import numpy as np 
import tensorflow as tf

tf.train.GradientDescentOptimizer 

X=tf.placeholder(dtype="float", name="X")
Y=tf.placeholder(dtype="float",name="Y")
W=tf.Variable(0.0,name="W")
    
y_pred=tf.mul(X, W)
    
cost=tf.square(Y-y_pred)
    
train_op=tf.train.GradientDescentOptimizer(0.01).minimize(cost)
trX=np.linspace(1, -1, 101)
trY= 2*trX+ np.random.randn(*trX.shape)*.33
    

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    for i in range(100):
        for (x,y) in zip(trX, trY):
            sess.run(train_op,feed_dict={X:x,Y:y})
    print sess.run(W)
            