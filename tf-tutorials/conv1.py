from __future__ import print_function
import tensorflow as tf
import imageio
import numpy as np
import matplotlib.pyplot as plt

from segmentation.carvana.data_reader_cardava import get_train_files_carvana_segmentation, image_height, image_width

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
    raw_img=imageio.imread(image_path)/255
    image_input = np.reshape(raw_img,[1,image_height, image_width,3])
    print(image_input.shape)
    X=tf.placeholder(tf.float32,shape=[None,image_height,image_width,3])
    filters= tf.truncated_normal([3,3,3,3],mean=1,stddev=0.5) *0.1
    conv1 = tf.nn.conv2d(X,filters,[1,1,1,1],'SAME')
    r=tf.nn.relu(conv1)
    #pool=tf.nn.max_pool(r,ksize=[2,2,],strides=[1,2,2,1],padding='SAME')
    decov1 = tf.nn.conv2d_transpose(conv1,filters,[1,image_height, image_width,3],[1,1,1,1],"SAME")

    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    out_image, decov1_out=sess.run([conv1,decov1],feed_dict={
                    X: image_input
                })
    amin=np.min(decov1_out)
    amax=np.max(decov1_out)

    decov1_out_r= 255*(decov1_out-amin)/(amax-amin)

    decov1_out_r =decov1_out_r.astype(np.uint8)
    out_image_r = out_image/np.max(out_image)

    print(np.max(decov1_out_r))
    print(np.min(decov1_out_r))

    fig = plt.figure()
    plt.subplot(2,3,1)
    plt.imshow(raw_img)
    plt.title("Input Image")
    #
    # print(out_image_r.shape)
    # plt.subplot(2, 3, 2)
    # plt.imshow(np.reshape(out_image_r,[image_height,image_width,3]))
    # plt.title("Conv1 Image")


    plt.subplot(2, 3, 3)
    ori_image=np.reshape(decov1_out_r,[image_height,image_width,3])
    plt.imshow(ori_image)
    plt.title("Deconv1 Image")

    plt.show()

    #print(image_input)
    #print(decov1_out_r)
    imageio.imwrite("or.jpg",ori_image)

    