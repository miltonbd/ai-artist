from __future__ import print_function
import tensorflow as tf
import cv2

class Padding:
    SAME='SAME'
    VALID='VALID'
    

def conv(X,filters,b,stride,padding=Padding.SAME):
    
    tf.nn.conv2d(X, filters, [1,stride,stride,1] , padding)
    
    X=tf.nn.bias_add(X, b)
    
    return tf.nn.relu(X)

def max_pool(X,k,stride,pad=Padding.SAME):
    return tf.nn.max_pool(X, [1,k,k,1], [1,stride,stride,1], pad)
    

with tf.Session() as sess:
    X=tf.placeholder('float', [None,448,448,3])
    
    print (tf.VERSION)
    
    
    