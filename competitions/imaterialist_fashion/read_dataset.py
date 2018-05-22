import os
import json

data_dir="/media/milton/ssd1/dataset/imaterialist_fashion"

train_json=os.path.join(data_dir,"train.json")

with open(train_json, 'r') as f:
    json_data = json.loads(f.read())
    print(len(json_data['images']))