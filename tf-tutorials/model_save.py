import numpy as np 
import tensorflow as tf 

x=tf.Variable(tf.random_normal((10,5), stddev=0.5, name="Weight"))
weight=tf.Variable(tf.random_normal((5,6), stddev=0.5, name="Weight"))
biases=tf.Variable(tf.random_normal(shape=(10,6), stddev=0.5, name="biases"))

value=tf.matmul(x, weight, name="value")+biases

saver=tf.train.Saver()

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    sess.run(value)
    
    #saver.save(sess, "models/model.ckpt")
    
    print "Model Saved"
    
    saver.restore(sess, "models/model.ckpt")
    
    print "Model Restored"
    
    
    