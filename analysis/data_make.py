import tifffile
import numpy as np
import os
import json
import collections
import random
from PIL import Image
import shutil

from tqdm import tqdm

import LDCTIQAG2023_train as data_path
import os.path as osp


def mosaic(new_dataset_size):
    img_path = osp.join(osp.dirname(data_path.__file__), 'image')
    label_path = osp.join(osp.dirname(data_path.__file__), 'train.json')
    save_path = os.path.join(osp.dirname(data_path.__file__), f"mosaic_dataset_{new_dataset_size // 1000}K")
    save_path_imgs = os.path.join(osp.dirname(data_path.__file__), f"mosaic_dataset_{new_dataset_size // 1000}K", "image")

    if not os.path.exists(save_path):
        os.mkdir(save_path)
        os.mkdir(save_path_imgs)

    for file_name in os.listdir(img_path):
        if "tif" in file_name:
            full_file_name = os.path.join(img_path, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, os.path.join(save_path_imgs))

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

    num = new_dataset_size // 20
    save_img_index = 1000
    for _ in tqdm(range(num)):
        mode = random.choice([2, 3, 4])
        for key in label_dict:
            files = label_dict[key]
            random.shuffle(files)
            img_tmp = np.zeros((512, 512))
            if mode == 2:
                img1 = img_dict[files[0]]
                img2 = img_dict[files[1]]
                choices = [1, 2]
                choice = random.choice(choices)
                if choice == 1:
                    value = random.randint(200, 300)
                    img_tmp[:, :value] = img1[:, :value]
                    img_tmp[:, value:] = img2[:, value:]
                elif choice == 2:
                    value = random.randint(200, 300)
                    img_tmp[:value, :] = img1[:value, :]
                    img_tmp[value:, :] = img2[value:, :]
            elif mode == 3:
                img1 = img_dict[files[0]]
                img2 = img_dict[files[1]]
                img3 = img_dict[files[2]]
                choices = [1, 2, 3, 4]
                choice = random.choice(choices)
                if choice == 1:
                    row_value = random.randint(200, 300)
                    col_value = random.randint(200, 300)
                    img_tmp[:row_value, :col_value] = img1[:row_value, :col_value]
                    img_tmp[:row_value, col_value:] = img2[:row_value, col_value:]
                    img_tmp[row_value:, :] = img3[row_value:, :]
                elif choice == 2:
                    row_value = random.randint(200, 300)
                    col_value = random.randint(200, 300)
                    img_tmp[:row_value, :] = img1[:row_value, :]
                    img_tmp[row_value:, :col_value] = img2[row_value:, :col_value]
                    img_tmp[row_value:, col_value:] = img3[row_value:, col_value:]
                elif choice == 3:
                    row_value = random.randint(200, 300)
                    col_value = random.randint(200, 300)
                    img_tmp[:, :col_value] = img1[:, :col_value]
                    img_tmp[:row_value, col_value:] = img2[:row_value, col_value:]
                    img_tmp[row_value:, col_value:] = img3[row_value:, col_value:]
                elif choice == 4:
                    row_value = random.randint(200, 300)
                    col_value = random.randint(200, 300)
                    img_tmp[:, col_value:] = img1[:, col_value:]
                    img_tmp[:row_value, :col_value] = img2[:row_value, :col_value]
                    img_tmp[row_value:, :col_value] = img3[row_value:, :col_value]
            elif mode == 4:
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
            img = Image.fromarray(img_tmp)
            img.save(os.path.join(save_path_imgs, f"{save_img_index}.tif"))
            data[f"{save_img_index}.tif"] = key
            save_img_index += 1

    with open(os.path.join(save_path, 'data.json'), 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == '__main__':
    mosaic(new_dataset_size=10000)