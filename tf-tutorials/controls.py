import tensorflow as tf 

a=tf.constant(5)
b=tf.constant(6)

def a1():
    return tf.mul(a, b)
    
def b1():
    return tf.div(a, b)

cond1=tf.cond(tf.less(a, b), a1,b1 )


i=tf.placeholder(tf.int32)

j= tf.placeholder(tf.int32)

sum= tf.Variable(0)

def loop_cond1(i,j,sum):
    
    return i<j

def body1(i,j,sum):
    i=tf.add(i,1)
    j=tf.add(j,1)
    sum=tf.add(i,1)
    return [i,j,sum]

loop1=tf.while_loop(cond=loop_cond1, body=body1, loop_vars=[i,j,sum])

with tf.Session() as sess:
    
    tf.global_variables_initializer()
    
    print sess.run(loop1)
    
    print sess.run(loop1)
    
    