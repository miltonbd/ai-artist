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


class Augment:
    pass

if __name__ == '__main__':
    resizeAll(train, 224, "classification_train","ISIC-2017_Training_Data")
    #resizeAll(valid, 224, "classification_valid", "ISIC-2017_Validation_Data")
    #resizeAll(test, 224, "classification_test", "ISIC-2017_Test_v2_Data")

