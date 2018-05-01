

"""
Base classes for All model params.
must be tensortflow or pytorch agnostic
"""

class BaseModeParams(object):
    def __init__(self):
        self.num_gpus=1
        self.epochs = 200
        self.dropout=0.6
        self.batch_size_train_per_gpu =10
        self.batch_size_train_per_gpu = 50
        self.num_classes = 2
        self.learning_rate = 0.001
        self.log_dir = 'logs'
