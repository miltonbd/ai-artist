import tensorflow as tf
from utils.TensorflowUtils import variable_on_cpu, variable_with_weight_decay

class MYUnet(object):
    def __init__(self,image_height, image_width, channels, num_classes):
        self.image_height=image_height
        self.image_width=image_width
        self.channels=channels
        self.num_classes=num_classes

    def build(self,input_batch):

        









