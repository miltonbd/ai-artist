import os
import PIL
from PIL import Image
import csv

data_dir = "/home/milton/dataset/skin/"

gt = os.path.join(data_dir,"ISIC-2017_Training_Part3_GroundTruth.csv")

def resizeAll(size):
    with open(gt, newline='\n') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        i=0
        melanomas=[]
        seborrheic_keratosis=[]
        nevus=[]
        for line in spamreader:
            if i > 0 :
                image_path = os.path.join(data_dir,'ISIC-2017_Training_Data', line[0]+'.jpg')
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
        print("total images {}".format(i-1))
        print("melanomas {}".format(len(melanomas)))
        print("seborrheic_keratosis {}".format(len(seborrheic_keratosis)))
        print("nevus {}".format(len(nevus)))
        resize(size,melanomas,'melanomas')
        resize(size,seborrheic_keratosis,'seborrheic_keratosis')
        resize(size,nevus,'nevus')


def resize(size,paths,class_folder):
    for path in paths:
        img = Image.open(path)
        img = img.resize((size, size), PIL.Image.ANTIALIAS)
        save_dir = os.path.join(data_dir,'classification_{}'.format(size),class_folder)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        path, filename = os.path.split(path)
        save_path = os.path.join(save_dir, filename)
        img.save(save_path)


class Augment:
    pass

if __name__ == '__main__':
    resizeAll(224)
    print("finished")