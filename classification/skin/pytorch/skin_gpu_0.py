import os
from classification.skin.pytorch.skin_classifier import SkinLeisonClassfication
from classification.models.pytorch.pretrainedmodels.models.vggm import vggm


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

        self.model_name = vggm
        self.model_log_name="adam2"
        self.learning_rate =  0.01
        self.model_name_str="vgg_gpu0"

model=Model()

clasifier=SkinLeisonClassfication('logs/adam2')
clasifier.load_data()
clasifier.load_model(model)
for epoch in range(clasifier.start_epoch, clasifier.start_epoch + clasifier.epochs):
    try:
      clasifier.train(epoch)
      clasifier.test(epoch)
    except KeyboardInterrupt:
      clasifier.test(epoch)
      break;


