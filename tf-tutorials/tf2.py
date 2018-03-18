import tensorflow as tf 
import numpy as np

x=tf.Variable(tf.random_normal((2,4), stddev=0.1))

y=tf.Variable(tf.random_normal((2,4), stddev=0.1))

z=x+y

input1=tf.placeholder(tf.int32, shape=(2,3))

pad=tf.placeholder(tf.int32, shape=(2,2))

padded=tf.pad( input1, pad )

a=tf.placeholder(tf.int32, shape=(3))

b=tf.placeholder(tf.int32, shape=(3))

c=tf.placeholder(tf.int32, shape=(3))

d=tf.pack([a,b,c])
e=tf.pack([a,b,c],axis=1)

g=tf.placeholder(dtype=tf.float32)

h=g/2
reduce_max_input=tf.placeholder(dtype=tf.float32, shape=(7,7,2))

reduce_max=tf.reduce_max(reduce_max_input, axis=2, keep_dims=True)

reduce_mean_input=tf.placeholder(dtype=tf.float32, shape=(7,7,2))

reduce_mean=tf.reduce_mean(reduce_mean_input, axis=2)

reduce_sum_input=tf.placeholder(tf.float32,(7,7,2,4))
reduce_sum=tf.reduce_sum(reduce_sum_input, reduction_indices=[1])


max1=tf.placeholder(tf.float32,(7,7,2,4))
max2=tf.placeholder(tf.float32,(4,))
maxi=tf.maximum(max1, max2)

mini=tf.minimum(max1, max2)

tr=tf.transpose(max1,[1,2,3,0])

intinput=tf.placeholder(tf.int32, ( 7, 7, 2, 4 ))

cast1=tf.cast(intinput>100, dtype=tf.float32)

with tf.Session() as sess:
    out= sess.run(padded,feed_dict={input1:np.ones((2,3)),pad:np.asanyarray([[2,1],[2,1]], dtype=np.int32)})
    print  (out)
    
    out1= sess.run([d,e],feed_dict={a:np.asarray([1,2,3]),b:np.asarray([4,5,6]),c:np.asarray([7,8,9])})
    print out1[0]
    print out1[1]
    
    print sess.run(h, feed_dict={g:1.1})
    
    print np.tile(np.random.rand(7,7,1,4), [1,1,2,1]).shape
    
    print "Reduce Max ",sess.run(reduce_max,feed_dict={reduce_max_input:np.random.rand(7,7,2)}).shape
    
    print sess.run(reduce_mean,feed_dict={reduce_mean_input:np.random.rand(7,7,2)}).shape
    
    print sess.run(maxi,feed_dict={max1:np.random.rand(7,7,2,4), max2:np.random.rand(4,)}).shape
    
    print sess.run(mini,feed_dict={max1:np.random.rand(7,7,2,4), max2:np.random.rand(4,)}).shape
    
    print sess.run(tr,feed_dict={max1:np.random.rand(7,7,2,4)}).shape
    
    print sess.run(cast1, feed_dict={intinput:np.random.randint(448,size=(7,7,2,4))}).shape
    
    print sess.run(reduce_sum, feed_dict={reduce_sum_input:np.random.randint(448,size=(7,7,2,4))}).shape
    
    
    
    