import os


class BaseTrainer(object):

    def __init__(self, data_reader, model_params, model):
        self.data_reader = data_reader
        self.model_params = model_params
        self.num_gpus = model_params.num_gpus
        self.image_height = data_reader.image_height
        self.image_width = data_reader.image_width
        self.num_channels = data_reader.num_channels
        self.num_classes = data_reader.num_classes
        self.batch_size_train_per_gpu = model_params.batch_size_train_per_gpu
        self.batch_size_test_per_gpu = model_params.batch_size_test_per_gpu
        self.batch_size_train = self.batch_size_train_per_gpu * self.num_gpus
        self.batch_size_test = self.batch_size_test_per_gpu * self.num_gpus
        self.num_threads = data_reader.num_threads
        self.learning_rate = model_params.learning_rate
        self.epochs = model_params.epochs
        self.dropout = model_params.dropout
        self.optimizer = model_params.optimizer
        self.model = model
        self.logdir = model.logdir
        self.savedir = os.path.join(self.logdir, "saved_models/")

        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)

        """
        Adam: Auto learning rate decay with momentum

        """

    def showParamsBeforeTraining(self):
        print("num classes {}".format(self.num_classes))
        print("num gpus {}".format(self.num_gpus))
