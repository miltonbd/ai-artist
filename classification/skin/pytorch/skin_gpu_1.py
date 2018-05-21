import os
from torch import optim, nn
from classification.skin.pytorch.skin_classifier import SkinLeisonClassfication
from classification.models.pytorch.vgg import vgg19_bn

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# net = VGG('VGG19',num_classes)
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d(num_classes=2)
# net = MobileNet()
# net = MobileNetV2()
# net = ShuffleNetG2()
# net = SENet18()

class Model(object):
    """
    store all model and optmization related params here.
    """
    def __init__(self):
        model_conv=vgg19_bn()
        num_ftrs = model_conv.classifier[6].in_features

        # convert all the layers to list and remove the last one
        features = list(model_conv.classifier.children())[:-1]

        ## Add the last layer based on the num of classes in our dataset
        features.extend([nn.Linear(num_ftrs, 2)])

        ## convert it into container and add it to our model class.
        model_conv.classifier = nn.Sequential(*features)
        self.model_name = model_conv
        self.model_log_name="adam1"
        self.learning_rate =  0.005
        self.optimizer="adam"
        self.model_name_str="vgg_gpu1"

model=Model()

clasifier=SkinLeisonClassfication('logs/adam1')
clasifier.load_data()
clasifier.load_model(model)
for epoch in range(clasifier.start_epoch, clasifier.start_epoch + clasifier.epochs):
    try:
      clasifier.train(epoch)
      clasifier.test(epoch)
    except KeyboardInterrupt:
      clasifier.test(epoch)
      break;
    #clasifier.load_data()


