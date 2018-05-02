import os
from segmentation.carvana.car_segmentation import CarvanaSegmentation
from classification.models.pytorch.vgg import VGG
from segmentation.carvana.model_params import ModeParams

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Model(object):
    """
    store all model and optmization related params here.
    """
    def __init__(self):
        self.model_name = VGG
        self.model_log_name="adam1"
        self.model_params=None

model=Model()
model.model_params=ModeParams()
trainer=CarvanaSegmentation('logs/adam1')
trainer.load_data()
trainer.load_model(model)
for epoch in range(trainer.start_epoch, trainer.start_epoch + trainer.epochs):
    try:
      trainer.train(epoch)
      trainer.test(epoch)
    except KeyboardInterrupt:
      trainer.test(epoch)
      break;
    #clasifier.load_data()

