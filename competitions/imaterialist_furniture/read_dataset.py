import os
import json
import PIL
import requests
import  threading
from time import sleep
from PIL import Image
from urllib.request import urlopen
from multiprocessing import Pool
import urllib


data_dir="/media/milton/ssd1/dataset/imaterialist_furniture"

train_json=os.path.join(data_dir,"test.json")

def save_image(obj):
    url_open=""
    try:
        url_open=url = obj['url'][0]
        filename = os.path.join(data_dir, 'test', url.split("/")[-1])
        if os.path.exists(filename):
            return
        import urllib
        req = urllib.request.Request(url, headers={'User-Agent': "Mozilla/5.0 (X11; Linux x86_64; rv:10.0) Gecko/20100101 Firefox/10.0"})
        con = urllib.request.urlopen(req,timeout=600)
        img = Image.open(con)
        new_img = img.convert('RGB').resize((240, 240), Image.ANTIALIAS)
        new_img.save(filename, 'jpeg')


    except Exception as e:
        print(e)
        print(url_open)


with open(train_json, 'r') as f:
    json_data = json.loads(f.read())
    i=1
    for obj in json_data['images']:
        # p=Pool(10)
        # p.map(save_image,(obj,i))
        # i=i+1
        t=threading.Thread(target=save_image, args=(obj,))
        t.start()
        i=i+1
        sleep(.1)
        #break