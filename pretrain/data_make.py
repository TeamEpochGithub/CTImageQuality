import tifffile
import numpy as np
import os
import json
import collections
import random
from PIL import Image
import shutil

img_path = r"C:\Users\leo\Documents\CTImageQuality\LDCTIQAG2023_train\image"
label_path = r"C:\Users\leo\Documents\CTImageQuality\LDCTIQAG2023_train\train.json"
save_path = r"C:\Users\leo\Documents\CTImageQuality\pretrain\create_dataset"

if not os.path.exists(save_path):
    os.mkdir(save_path)
    os.mkdir(os.path.join(save_path, "imgs"))

img_dict = dict()
for file in os.listdir(img_path):
    if file.endswith('.tif'):
        with tifffile.TiffFile(os.path.join(img_path, file)) as tif:
            image = tif.pages[0].asarray()
            img = np.array(image)
            img_dict[file] = img

label_dict = collections.defaultdict(list)
with open(label_path) as file:
    data = json.load(file)
    for key in data:
        label_dict[data[key]].append(key)

imgs = []
labels = []
def mosaic(num):
    for i in range(num):
        mode = random.choice([2,3,4])
        for key in label_dict:
            files = label_dict[key]
            random.shuffle(files)
            img_tmp = np.zeros((512,512))
            if mode==2:
                img1 = img_dict[files[0]]
                img2 = img_dict[files[1]]
                choices = [1, 2]
                choice = random.choice(choices)
                if choice==1:
                    value = random.randint(200, 300)
                    img_tmp[:, :value] = img1[:, :value]
                    img_tmp[:, value:] = img2[:, value:]
                elif choice==2:
                    value = random.randint(200, 300)
                    img_tmp[:value, :] = img1[:value, :]
                    img_tmp[value:, :] = img2[value:, :]
            elif mode==3:
                img1 = img_dict[files[0]]
                img2 = img_dict[files[1]]
                img3 = img_dict[files[2]]
                choices = [1, 2, 3, 4]
                choice = random.choice(choices)
                if choice==1:
                    row_value = random.randint(200, 300)
                    col_value = random.randint(200, 300)
                    img_tmp[:row_value, :col_value] = img1[:row_value, :col_value]
                    img_tmp[:row_value, col_value:] = img2[:row_value, col_value:]
                    img_tmp[row_value:, :] = img3[row_value:, :]
                elif choice==2:
                    row_value = random.randint(200, 300)
                    col_value = random.randint(200, 300)
                    img_tmp[:row_value, :] = img1[:row_value, :]
                    img_tmp[row_value:, :col_value] = img2[row_value:, :col_value]
                    img_tmp[row_value:, col_value:] = img3[row_value:, col_value:]
                elif choice==3:
                    row_value = random.randint(200, 300)
                    col_value = random.randint(200, 300)
                    img_tmp[:, :col_value] = img1[:, :col_value]
                    img_tmp[:row_value, col_value:] = img2[:row_value, col_value:]
                    img_tmp[row_value:, col_value:] = img3[row_value:, col_value:]
                elif choice==4:
                    row_value = random.randint(200, 300)
                    col_value = random.randint(200, 300)
                    img_tmp[:, col_value:] = img1[:, col_value:]
                    img_tmp[:row_value, :col_value] = img2[:row_value, :col_value]
                    img_tmp[row_value:, :col_value] = img3[row_value:, :col_value]
            elif mode==4:
                img1 = img_dict[files[0]]
                img2 = img_dict[files[1]]
                img3 = img_dict[files[2]]
                img4 = img_dict[files[3]]
                row_value = random.randint(200, 300)
                col_value = random.randint(200, 300)
                img_tmp[:row_value, :col_value] = img1[:row_value, :col_value]
                img_tmp[row_value:, :col_value] = img2[row_value:, :col_value]
                img_tmp[:row_value, col_value:] = img3[:row_value, col_value:]
                img_tmp[row_value:, col_value:] = img4[row_value:, col_value:]
            imgs.append(img_tmp)
            labels.append(key)

mosaic(10)

for file_name in os.listdir(img_path):
    if "tif" in file_name:
        full_file_name = os.path.join(img_path, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, os.path.join(save_path, "imgs"))

start_num = 1000
for i in range(len(imgs)):
    img = Image.fromarray(imgs[i])
    img.save(os.path.join(save_path, "imgs", f"{start_num+i}.tif"))
    label = labels[i]
    data[f"{start_num+i}.tif"] = label

with open(os.path.join(save_path, 'data.json'), 'w') as f:
    json.dump(data, f, indent=4)

