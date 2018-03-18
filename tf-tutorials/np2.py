import numpy as np
import os

input1=np.random.rand(2,3,4)

print np.repeat(input1, 5, 0).shape

scale_vector = [[]]
print scale_vector.extend([2]*4)
print scale_vector.extend([1]*20)

print np.arange(7)


input2=np.random.random_integers(0,1,(5,6))

print np.nonzero(input2) # result of each dimension nth elemts across dimension is non zero

input3=np.ones((3,5))

print np.pad(input3, [[0,47],[0,1]], 'constant').shape

print np.pad(np.zeros((1,1)),[[0,0],[0,0]],'constant')

print np.reshape([np.arange(7)] * 7 * 2, (7,7,2))

print os.path.basename("mmn/nnnmm/a.jpg")

print 1/2
