import numpy as np
import torch
import torchvision
import tifffile
from PIL import Image
import LDCTIQAG2023_train as train_data
import json

import os.path as osp
import os

import analysis


def create_datasets(imgs_list, label_list, configs, final_train=False, patients_out=True, patient_ids_out=[3]):
    if final_train:
        return CT_Dataset(imgs_list, label_list, split="train", config=configs), None

    if patients_out:
        patient_ids = np.loadtxt(osp.join(osp.dirname(analysis.__file__), 'labels.txt'))
        patient_indices = [i for i, x in enumerate(patient_ids) if x in patient_ids_out]  # np.where(patient_ids == patient_ids_out)[0]
        non_patient_indices = list(set(list(range(1000))) - set(patient_indices))
        print(len(patient_indices), len(non_patient_indices))
        train_dataset = CT_Dataset([imgs_list[x] for x in non_patient_indices],
                                   [label_list[x] for x in non_patient_indices], split="train",
                                   config=configs)
        test_dataset = CT_Dataset([imgs_list[x] for x in patient_indices], [label_list[x] for x in patient_indices],
                                  split="test", config=configs)
    else:
        left_bound, right_bound = 100, 1000  # 900, 1000

        train_dataset = CT_Dataset(imgs_list[:left_bound] + imgs_list[right_bound:],
                                   label_list[:left_bound] + label_list[right_bound:], split="train", config=configs)
        test_dataset = CT_Dataset(imgs_list[left_bound:right_bound], label_list[left_bound:right_bound], split="test",
                                  config=configs)
    return train_dataset, test_dataset


def create_datalists():
    data_dir = osp.join(osp.dirname(train_data.__file__), 'image')
    label_dir = osp.join(osp.dirname(train_data.__file__), 'train.json')
    with open(label_dir, 'r') as f:
        label_dict = json.load(f)

    imgs_list = []
    label_list = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.tif'):
                label_list.append(label_dict[file])
                with tifffile.TiffFile(os.path.join(root, file)) as tif:
                    image = tif.pages[0].asarray()
                    img = Image.fromarray(image)
                    imgs_list.append(img)

    return imgs_list, label_list


class CT_Dataset(torch.utils.data.Dataset):
    def __init__(self, imgs_list, label_list, split='validation', config={'img_size': 512}):
        self.imgs_list = imgs_list
        self.label_list = label_list
        self.split = split
        self.image_size = config['img_size']
        self.config = config

        if self.split == 'train':

            operations = [torchvision.transforms.ToPILImage()]

            if self.config['RandomHorizontalFlip']:
                operations.append(torchvision.transforms.RandomHorizontalFlip())

            if self.config['RandomVerticalFlip']:
                operations.append(torchvision.transforms.RandomVerticalFlip())

            if self.config['RandomRotation']:
                operations.append(torchvision.transforms.RandomRotation(self.config['rotation_angle']))

            if self.config['ZoomIn']:
                operations.append(torchvision.transforms.RandomApply([
                    torchvision.transforms.CenterCrop(size=int(self.config['zoomin_factor'] * 512)),
                    torchvision.transforms.Resize(size=(self.image_size, self.image_size)),
                ], p=0.1))
            if self.config['ZoomOut']:
                operations.append(torchvision.transforms.RandomApply([
                    torchvision.transforms.Pad(padding=int(512 * self.config['zoomout_factor'])),
                    torchvision.transforms.Resize((self.image_size, self.image_size)),
                ], p=0.1))

            if self.config['XShift'] or self.config['YShift']:
                x_max_shift = np.random.uniform(low=0.0, high=self.config['max_shift']) if self.config['XShift'] else 0
                y_max_shift = np.random.uniform(low=0.0, high=self.config['max_shift']) if self.config['YShift'] else 0
                shifts = (x_max_shift, y_max_shift)
                operations.append(torchvision.transforms.RandomApply([
                    torchvision.transforms.RandomAffine(degrees=0, translate=shifts)
                ], p=0.1))

            if self.config['RandomShear']:
                shear_degree = np.random.uniform(low=0.0, high=self.config['max_shear'])
                operations.append(torchvision.transforms.RandomApply([
                    torchvision.transforms.RandomAffine(degrees=0, translate=shear_degree)
                ], p=0.1))

            operations += [torchvision.transforms.ToTensor()]

            self.transform = torchvision.transforms.Compose(operations)

        else:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        x = self.imgs_list[idx]
        x = x.resize((self.image_size, self.image_size), Image.ANTIALIAS)
        x = np.array(x)
        x = self.transform(x)
        y = self.label_list[idx]

        return x, torch.tensor(y)
