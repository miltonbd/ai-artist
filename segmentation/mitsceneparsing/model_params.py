import tensorflow as tf

class ModeParams(object):
    def __init__(self):
        self.num_gpus=1
        self.num_gpus= 1
        self.epochs = 200
        self.learning_rate = 1e-3
        self.dropout=0.6
        self.batch_size_test_per_gpu  =2
        self.batch_size_train_per_gpu =2
        self.optimizer= tf.train.AdagradOptimizer(self.learning_rate)
