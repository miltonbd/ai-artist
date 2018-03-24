import imageio
import tensorflow as tf
import matplotlib
from utils.data_reader_cifar10 import DataReaderCifar10
import matplotlib.pyplot as plt

data_reader = DataReaderCifar10(10,2,1)
data_reader.loadDataSet()
images, labels = data_reader.nextBatch()
img=images[3]
print(img.dtype)
imageio.imwrite("a.jpg",img)
