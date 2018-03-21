import tensorflow as tf 
import numpy as np
import os

input=tf.constant([[.6,.8,.9],[1,-3,.7]])
weights=tf.constant([[-.5,.6],[.8,.8],[1, -1]])
biases=tf.constant([[1, -1], [ -1, 1]])
outputs=tf.matmul(input,weights) 
tf.summary.scalar("scalar_summary", outputs)


x=tf.placeholder(tf.float32,1)
y=tf.placeholder(tf.float32,1)
z=tf.placeholder(tf.float32,1)

select1=tf.where(x>y, y, z )

greater1=tf.greater(x, y )

with tf.Session() as sess:
    tf.initialize_all_variables()
    c=tf.constant(value="Hello World")
    print (sess.run(c))
    
    print (sess.run(outputs))
    
    a=tf.random_normal((7,7,2))
    b=tf.random_normal((7,2,8))
    c=tf.matmul(a, b)
    
    a1=tf.random_normal((7,7,2,5))
    b1=tf.random_normal((1,1,2,5))
    d= tf.multiply(a1, b1)

    print (sess.run(c).shape)
    
    print (sess.run(d).shape)
    
    print (sess.run(select1, feed_dict={x:[1],y:[2],z:[5]}))
    
    print (sess.run(greater1, feed_dict={x:[1],y:[2],z:[5]}))
    
    concat_i1=tf.ones((2,2,2)) * 2
    
    concat_i2=tf.ones((2,2,2))

    images = tf.random_normal(shape=[10,224,224,3], name="images")
    b,g,r = tf.split(images, 3, axis=3)
    rgb = tf.concat(values=[r,g,b], axis=3)
    print(sess.run(rgb).shape)

    aaa= tf.truncated_normal(shape=[1,2,5,1,8,1])
    print(sess.run(tf.squeeze(aaa)).shape)




    
    