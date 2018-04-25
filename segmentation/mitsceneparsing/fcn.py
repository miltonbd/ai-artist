from segmentation.base_segmentation_singe_gpu import BaseSegmentation
from segmentation.models.FCN import FCN
from segmentation.mitsceneparsing.data_reader_mitsceneparsing import DataReaderMitSceneParsing
from segmentation.mitsceneparsing.model_params import ModeParams

class MyMitSceneSegmentation(BaseSegmentation):
    def __init__(self, data_reader, model_params, model):
        BaseSegmentation.__init__(self,  data_reader, model_params, model)



if __name__ == '__main__':
    model_params = ModeParams()
    model_params.num_gpus=2
    data_reader=DataReaderMitSceneParsing()
    model=FCN(data_reader)
    model.logdir = 'logs_dir'

    segmentor=MyMitSceneSegmentation( data_reader, model_params, model)

    segmentor.train()
