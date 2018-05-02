from segmentation.base_segmentation_singe_gpu import BaseSegmentation
from segmentation.models.tensorflow.FCN import FCN
from segmentation.mitsceneparsing.data_reader_mitsceneparsing import DataReaderMitSceneParsing

class MyCarvanaSegmentation(BaseSegmentation):
    def __init__(self,params):
        BaseSegmentation.__init__(self, params)



if __name__ == '__main__':
    params={
        'batch_size_train_per_gpu':50,
        'num_gpus':1,
        'epochs':200,
        'image_height': 224,
        'image_width': 224,
        'num_classes':1,
        'model': FCN(image_height=224, image_width=224, num_classes=151, channels=3),
        'data_reader' : DataReaderMitSceneParsing()
    }

    segmentor=MyCarvanaSegmentation(params)
    params_training = {
        'logdir': 'cifar10_multi_log/'
    }
    segmentor.train(params_training)
