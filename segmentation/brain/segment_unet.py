from segmentation.base_classifier import BaseClassifier
from segmentation.models.u_net_tf import UNet

class MyCarvanaSegmentation(BaseClassifier):
    def __init__(self,params):
        BaseClassifier.__init__(self,params)



if __name__ == '__main__':
    params={
        'batch_size_train_per_gpu':50,
        'num_gpus':1,
        'epochs':200,
        'image_height': 1280,
        'image_width': 1918,
        'num_classes':1,
        'model': UNet()
    }

    segmentor=MyCarvanaSegmentation(params)
    params_training = {
        'logdir': 'cifar10_multi_log/'
    }
    segmentor.train(params_training)
