import os.path as osp
import os
import json
import tifffile
import random
from PIL import Image
import numpy as np

random.seed(42)

data_path = r"C:\Users\leo\Documents\CTImageQuality\LDCTIQAG2023_train"
output_path = r"C:\Users\leo\Documents\CTImageQuality\pretrain\weighted_dataset"

imgs = osp.join(data_path, 'image')
labels = osp.join(data_path, 'train.json')
with open(labels, 'r') as f:
    label_dict = json.load(f)

imgs_list = []
label_list = []
for root, dirs, files in os.walk(imgs):
    for file in files:
        if file.endswith('.tif'):
            label_list.append(label_dict[file])
            with tifffile.TiffFile(os.path.join(root, file)) as tif:
                image = tif.pages[0].asarray()
                img = Image.fromarray(image)
                img = np.float32(np.array(img))
                imgs_list.append(img)

nums = 10000
new_label_dict = dict()
for i in range(nums):
    index1 = random.randint(0, len(imgs_list) - 1)
    index2 = random.randint(0, len(imgs_list) - 1)
    weights = random.uniform(0.25, 0.75)
    new_img = imgs_list[index1] * weights + imgs_list[index2] * (1 - weights)
    new_label = int(label_list[index1] * weights + label_list[index2] * (1 - weights))
    if new_label == 4:
        new_label = 3
    np.save(osp.join(output_path, "image", f"{i}.npy"), new_img)
    new_label_dict[f"{i}.npy"] = new_label

with open(osp.join(output_path, 'label.json'), 'w') as json_file:
    json.dump(new_label_dict, json_file, indent=4)
