import tensorflow as tf 

with tf.variable_scope("a"):
    v1=tf.get_variable("v1",[1,1])
    

with tf.variable_scope("b"):
    v1=tf.get_variable("v1",(5,5))
    

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("b", reuse=True):
        print tf.get_variable("v1")
    