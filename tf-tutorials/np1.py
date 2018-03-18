import numpy as np

input1=np.random.random_integers(1,5,(3,5))

print np.expand_dims(input1, axis=1).shape

print np.reshape(input1, (3,1,5)).shape

print input1

print np.argmax(input1,1)

print -1*float('inf')

print np.random.randint(1,5,3)

print np.random.random_integers(1,5,3)

print np.random.choice(5,(2,2))

names="This is Bangladesh".split()  # if split has argument it will be used as delimiter. space is default

print "#".join(names)

print np.squeeze(np.random.rand(5,5,2)).shape

# makes any number dimension matrix to one dimension array
print np.random.rand(5,6,7,2).flatten().shape

