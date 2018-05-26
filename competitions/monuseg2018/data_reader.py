from PIL import Image
import os
import glob
import numpy as np
import zlib
import imageio
import nibabel
import  cv2
import xml.etree.ElementTree as ET


data_dir="/media/milton/ssd1/dataset/competitions/MoNuSeg/MoNuSeg_Training_Data"


def get_all_train_files():
    # train_images=[]
    # for file_path in glob.glob(os.path.join(data_dir,"Tissue_images","**tif")):
    #     train_images.append(np.asarray(cv2.imread(file_path)).reshape(-1))
    # train_images=np.asarray(train_images).reshape(-1, 1000, 1000, 3)

    breast_carcinoma=['TCGA-A7-A13E-01Z-00-DX1',
                      'TCGA-A7-A13F-01Z-00-DX1',
                      'TCGA-AR-A1AK-01Z-00-DX1',
                      'TCGA-AR-A1AS-01Z-00-DX1',
                      'TCGA-E2-A1B5-01Z-00-DX1',
                      'TCGA-E2-A14V-01Z-00-DX1']

    kidney_carcinoma=['TCGA-B0-5711-01Z-00-DX1',
                      'TCGA-HE-7128-01Z-00-DX1',
                      'TCGA-HE-7129-01Z-00-DX1',
                      'TCGA-HE-7130-01Z-00-DX1',
                      'TCGA-B0-5710-01Z-00-DX1',
                      'TCGA-B0-5698-01Z-00-DX1'
                      ]
    liver_carcinoma=[
                        'TCGA-18-5592-01Z-00-DX1',
                        'TCGA-38-6178-01Z-00-DX1',
                        'TCGA-49-4488-01Z-00-DX1',
                        'TCGA-50-5931-01Z-00-DX1',
                        'TCGA-21-5784-01Z-00-DX1',
                        'TCGA-21-5786-01Z-00-DX1']

    bladder_Bladder_Urothelia_Carcinoma=['TCGA-DK-A2I6-01A-01-TS1','TCGA-G2-A2EK-01A-02-TSB']

    colon_adenocarcinoma=['TCGA-AY-A8YK-01A-01-TS1','TCGA-NH-A8F7-01A-01-TS1']

    prostrate_adenocarcinoma=[
        'TCGA-G9-6336-01Z-00-DX1',
        'TCGA-G9-6348-01Z-00-DX1',
        'TCGA-G9-6356-01Z-00-DX1',
        'TCGA-G9-6363-01Z-00-DX1',
        'TCGA-CH-5767-01Z-00-DX1',
        'TCGA-G9-6362-01Z-00-DX1']

    stomach_adenocarcinoma=['TCGA-KB-A93J-01A-01-TS1',
                       'TCGA-RD-A8N9-01A-01-TS1']

    train_annotations=[]
    region=0
    i=0
    for file_path in glob.glob(os.path.join(data_dir,"Annotations","**xml")):
        xml_read=ET.parse(file_path)
        train_annotations.append(xml_read)
        for anotation in xml_read.getroot().findall('Annotation'):
            for regions in anotation.findall('Regions'):
                    region+=len(regions.findall('Region'))
        i+=1


    print(region)
    print(i)
    return








get_all_train_files()