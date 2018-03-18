import tensorflow as tf 
import numpy as np

a=tf.placeholder(tf.float32 )
b=tf.constant(2, tf.float32)
c=a//b # returns integer floor 
d=tf.placeholder(tf.float32)

temp=np.zeros((5,5))
ii=tf.Variable(0)
def scan1(previous_output, current_input):
    
    temp[ii,0]=1
    
    return previous_output
    
elems=np.arange(25)   

elems = tf.identity(elems)

initializer = tf.zeros((5,5),dtype=tf.int32)

sum=tf.scan(scan1, elems,initializer=initializer)
#  indices = tf.constant([[i,0]],dtype=tf.int64)  # A list of coordinates to update.

#    values = [5.0]  # A list of values corresponding to the respective
                # coordinate in indices.
#    shape = [5, 5]  # The shape of the corresponding dense tensor, same as `c`.

#    delta = tf.SparseTensor(indices, values, shape)
#    out=result-tf.sparse_tensor_to_dense(delta)

def body(x,y):
    return x+1,y

def condition(x,y):
    return x < 100

x = tf.Variable(0)
y=tf.Variable(0)

rows=tf.constant(5)
cols=tf.constant(5)
output=np.zeros((5,5))
input=np.zeros((5,5))

def body2(i,input1):
    v=input[0,0]
    output[0,0]=v
    return i+1, output
    

def cond2(i,input):
    
    return rows>i


loop2=tf.while_loop(cond2, body2,  [tf.constant(0),np.zeros((5,5))])




with tf.Session() as sess:
    tf.global_variables_initializer().run()
    
    print sess.run(c,feed_dict={a:4})
    
    print "sum ", sess.run(sum[-1])
    
    result = tf.while_loop(condition, body, [x,y])
    print sess.run(result)
    
    data=tf.ones((5,5))
    sh= tf.shape(data)
    m=tf.range(sh[1])*5
    
    print sess.run(m)
    
    #http://stackoverflow.com/questions/37697747/typeerror-tensor-object-does-not-support-item-assignment-in-tensorflow
    print sess.run(loop2[-1])
    
 
    
    
        
    
    # 
    
                  
    
    
    
    
    