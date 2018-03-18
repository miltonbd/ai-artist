import numpy as np 

a=np.asarray([
    [[1,2,3],[4,5,6]],
    [[1,2,3],[4,5,6]],
    [[1,2,3],[4,5,6]],
    [[1,2,3],[4,5,6]]
    ])
print np.random.rand(2,3,4,5).transpose().shape
print a.shape
print a.sum(axis=0)
print a.sum(axis=1)
print a.sum(axis=2)

data=np.random.randn(2,3,4)
print data.shape 
print np.reshape(data, [-1]).shape

print [2,2]*[[1],[1]]