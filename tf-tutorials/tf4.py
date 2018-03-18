import numpy as np 
import tensorflow as tf 
import cv2
import lasagne
import keras 


def aa(i):
     sum=0
     for i in np.arange(i):
         sum+=i
     return sum
 
aaa=tf.constant(0)+[aa(i) for i in np.arange(1000) ]

input=tf.ones((1125,1125))
val=input[0,0]

with tf.Session() as sess:
    
    print sess.run(aaa)
    print sess.run(val)