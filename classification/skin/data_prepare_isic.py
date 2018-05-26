import os
import PIL
from PIL import Image
import csv
import glob
import threading
from concurrent.futures import ThreadPoolExecutor
data_dir = "/home/milton/dataset/skin/"

train = os.path.join(data_dir,"ISIC-2017_Training_Part3_GroundTruth.csv")
valid = os.path.join(data_dir,"ISIC-2017_Validation_Part3_GroundTruth.csv")
test = os.path.join(data_dir,"ISIC-2017_Test_v2_Part3_GroundTruth.csv")

def resizeAll(csv_file,size,save_dir_prefix, source_dir):
    with open(csv_file, newline='\n') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        i=0
        melanomas=[]
        seborrheic_keratosis=[]
        nevus=[]
        for line in csvreader:
            if i > 0 :
                image_path = os.path.join(data_dir,source_dir, line[0]+'.jpg')
                if line[1]=='1.0':
                    melanomas.append(image_path)
                else:
                    if line[2] == '1.0':
                        seborrheic_keratosis.append(image_path)
                    else:
                        nevus.append(image_path)
            else:
                print(line)
            i+=1
        if 'train' in save_dir_prefix:
            for train_isic_add in glob.glob("/home/milton/dataset/skin/ISIC-images_melanoma/images/**"):
                melanomas.append(train_isic_add)

            for train_isic_add in glob.glob("/home/milton/dataset/skin/ISIC-images_sk/images/**"):
                seborrheic_keratosis.append(train_isic_add)

            for train_isic_add in glob.glob("/home/milton/dataset/skin/ISIC-images_nevus/images/**"):
                nevus.append(train_isic_add)
                #print(train_isic_add)


        print("total images {}".format(i-1))
        print("melanomas {}".format(len(melanomas)))
        print("seborrheic_keratosis {}".format(len(seborrheic_keratosis)))
        print("nevus {}".format(len(nevus)))
        resize(size,melanomas,'melanomas',save_dir_prefix)
        resize(size,seborrheic_keratosis,'seborrheic_keratosis',save_dir_prefix)
        resize(size,nevus,'nevus',save_dir_prefix)

def doResize(size,path,class_folder, save_dir_prefix):
    img = Image.open(path)
    img = img.resize((size, size), PIL.Image.ANTIALIAS)
    save_dir = os.path.join(data_dir, '{}_{}'.format(save_dir_prefix, size), class_folder)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    path, filename = os.path.split(path)
    save_path = os.path.join(save_dir, filename)
    img.save(save_path)
    img=[]


def resize(size,paths,class_folder, save_dir_prefix):
    for path in paths:
        #doResize(size,path,class_folder, save_dir_prefix)

        #executor = ThreadPoolExecutor(max_workers=5)
        #a = executor.submit(doResize,size,path,class_folder, save_dir_prefix)
        t = threading.Thread(target=doResize, args=(size,path,class_folder, save_dir_prefix))
        t.start()


import pandas as pd
import os

def move_ph2_images():
    excel=pd.read_excel('/media/milton/ssd1/dataset/skin/PH2Dataset/PH2_dataset.xlsx')
    nevus=list(excel.iloc[11:91, 0])
    nevus.extend(list(excel.iloc[92:171, 0]))
    melnomas=excel.iloc[172:211, 0]

    for id in melnomas:
        read_file=os.path.join("/media/milton/ssd1/dataset/skin/PH2Dataset/PH2 Dataset images/", id,"{}_Dermoscopic_Image".format(id),id+".bmp")

        save_file=os.path.join("/media/milton/ssd1/dataset/skin/classification_train_224/melanomas",id+".jpg")
        img=PIL.Image.open(read_file)
        img = img.resize((224, 224), PIL.Image.ANTIALIAS)
        img.save(save_file)
    for id in nevus:
        read_file = os.path.join("/media/milton/ssd1/dataset/skin/PH2Dataset/PH2 Dataset images/", id,
                                 "{}_Dermoscopic_Image".format(id), id + ".bmp")

        save_file = os.path.join("/media/milton/ssd1/dataset/skin/classification_train_224/nevus", id + ".jpg")
        img = PIL.Image.open(read_file)
        img = img.resize((224, 224), PIL.Image.ANTIALIAS)
        img.save(save_file)


def move_isic_2018_data():
    images_dir="/media/milton/ssd1/dataset/skin/ISIC2018_Task3_Training_Input"
    gt=os.path.join("/media/milton/ssd1/dataset/skin/ISIC2018_Task3_Training_GroundTruth","ISIC2018_Task3_Training_GroundTruth.csv")
    with open(gt, newline='\n') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        i = 0
        melanomas = []
        seborrheic_keratosis = []
        nevus = []
        for line in csvreader:
            if i > 0:
                image_path = os.path.join(data_dir, images_dir, line[0] + '.jpg')

                if line[1] == '1.0':
                    melanomas.append(image_path)
                else:
                    if line[2] == '1.0':
                        nevus.append(image_path)
                    else:
                        if line[5] == '1.0':
                            seborrheic_keratosis.append(image_path)
            else:
                print(line)
            i += 1
        print("melanomas: {}".format(len(melanomas)))
        print("nevus: {}".format(len(nevus)))
        print("sk: {}".format(len(seborrheic_keratosis)))

        for path in melanomas:
            save_file = os.path.join("/media/milton/ssd1/dataset/skin/classification_train_224/melanomas", path.split("/")[-1])
            img = PIL.Image.open(path)
            img = img.resize((224, 224), PIL.Image.ANTIALIAS)
            img.save(save_file)

        for path in nevus:
            save_file = os.path.join("/media/milton/ssd1/dataset/skin/classification_train_224/nevus",  path.split("/")[-1])
            img = PIL.Image.open(path)
            img = img.resize((224, 224), PIL.Image.ANTIALIAS)
            img.save(save_file)

        for path in seborrheic_keratosis:
            save_file = os.path.join("/media/milton/ssd1/dataset/skin/classification_train_224/seborrheic_keratosis",  path.split("/")[-1])
            img = PIL.Image.open(path)
            img = img.resize((224, 224), PIL.Image.ANTIALIAS)
            img.save(save_file)



if __name__ == '__main__':
    move_isic_2018_data()
    #move_ph2_images()
    #resizeAll(train, 224, "classification_train","ISIC-2017_Training_Data")
    #resizeAll(valid, 224, "classification_valid", "ISIC-2017_Validation_Data")
    #resizeAll(test, 224, "classification_test", "ISIC-2017_Test_v2_Data")

