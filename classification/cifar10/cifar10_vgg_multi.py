from classification.base_classifier import BaseClassifier
from classification.models.simplenetmultigpu import SimpleModel
from classification.cifar10.cifar10_model_params import Cifar10ModelParams
from classification.cifar10.data_reader_cifar10 import DataReaderCifar10

class MyCifar10(BaseClassifier):
    def __init__(self, data_reader, model_params, model):
        BaseClassifier.__init__(self,  data_reader, model_params, model)

if __name__ == '__main__':
    model_params = Cifar10ModelParams()
    model_params.num_gpus=2
    model_params.batch_size_train_per_gpu=100
    data_reader=DataReaderCifar10(model_params)
    model=SimpleModel(data_reader)
    model.logdir = 'logs_dir'
    trainer=MyCifar10(data_reader, model_params, model)

    trainer.train()
