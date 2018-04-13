import tensorflow as tf
import numpy as np


a=np.random.randint(0,9,[20,10])
a_tf = tf.stack(a)
tf.InteractiveSession()
print(a_tf.eval())

