import numpy as np 
import tensorflow as tf 
import os 

root= os.path.abspath("../")
print os.path.join(root, "my","ok")


data=np.random.randn(5,10)

print np.argmax(data,axis=1)

x=10 
print raw_input("enter some value: ")