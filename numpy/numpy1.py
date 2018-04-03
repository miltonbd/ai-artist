import numpy as np

randidx = np.random.randint(1000, size=[3,3,3,3])
print(randidx)
print(randidx.flatten())
print(randidx.ravel())