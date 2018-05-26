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

        # Let's freeze the same as above. Same code as above without the print statements
        num_ftrs = model_conv.classifier[6].in_features

        # convert all the layers to list and remove the last one
        features = list(model_conv.classifier.children())[:-1]

        ## Add the last layer based on the num of classes in our dataset
        features.extend([nn.Linear(num_ftrs, 2)])

        ## convert it into container and add it to our model class.
        model_conv.classifier = nn.Sequential(*features)
        self.model_name = model_conv
        self.learning_rate =  0.0001
        self.eps=1
        self.optimizer="adam"
        self.model_name_str="vgg_19bn"
        self.batch_size_train_per_gpu = 100
        self.batch_size_test_per_gpu = 2
        self.epochs = 100
        self.num_classes = 2
        self.logs_dir="logs/vgg19_bn"

model=Model()

clasifier=SkinLeisonClassfication(model)
clasifier.load_data()
clasifier.load_model()
for epoch in range(clasifier.start_epoch, clasifier.start_epoch + model.epochs):
    if epoch<10:
        clasifier.model.model_name.freeze_features_layers()
    else:
        clasifier.model.model_name.freeze_features_n_layers(30)
    try:
      clasifier.train(epoch)
      clasifier.test(epoch)
    except KeyboardInterrupt:
      clasifier.test(epoch)
      break;
    clasifier.load_data()


