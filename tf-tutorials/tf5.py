import tensorflow as tf 

x=tf.Variable(tf.random_normal((5,6)))
x_op=tf.assign(x,tf.random_normal((5,6)))

with tf.Session() as sess:
    print (x_op.eval())
    
    

with tf.Graph().as_default():
    print ("Default graph")