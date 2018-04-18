from __future__ import print_function
import tensorflow as tf
import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt

from utils.data_reader_cardava import get_train_files_carvana_segmentation, image_height, image_width

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
    images,_ = get_train_files_carvana_segmentation()
    image_path=images[100]
    print(image_path)
    image_input = np.reshape(imageio.imread(image_path),[1,image_height, image_width,3])
    print(image_input.shape)
    X=tf.placeholder(tf.float32,shape=[None,image_height,image_width,3])
    filters= tf.truncated_normal([3,3,3,3])
    conv1 = tf.nn.conv2d(X,filters,[1,1,1,1],'SAME')
    r=tf.nn.relu(conv1)
    decov1 = tf.nn.conv2d_transpose(r,filters,[1,image_height, image_width,3],[1,1,1,1],"SAME")
    decov1_r = tf.nn.relu(decov1)

    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    out_image, decov1_out=sess.run([conv1,decov1_r],feed_dict={
                    X: image_input
                })
    decov1_out_r= decov1_out/ np.max(decov1_out)
    plt.imshow(np.reshape(decov1_out_r,[image_height,image_width,3]))
    plt.show()
    print(out_image.shape)
    
    