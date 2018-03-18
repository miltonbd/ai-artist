from __future__ import print_function, division
import tensorflow as tf 
import numpy as np 


cond1=lambda j: j<10
body1=lambda j:  20
out=tf.while_loop(cond1, body1,[tf.constant(0)])


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    re=sess.run(out)   
    print (re)      
    
             

print( np.floor(1.8))
print ( np.ceil(1.8))
for i, x in enumerate(np.random.randn(10)):
    print (i," ",x)
    
a=np.random.randn(5,5)
print ( np.lib.pad(np.ones((2,2)), (1,2), mode="constant",constant_values=(4, 6)) )

print ( np.tile([1,2], (2,2)))

print ( np.zeros_like(a))

print ( np.zeros((2,2)) )  

d=np.random.randn(1,16,7,7,30)
s=d.shape
print ( np.reshape(d, (s[1],s[2],s[3],s[4])).shape )

data=np.asarray([1,2])

print (data[0:1])







