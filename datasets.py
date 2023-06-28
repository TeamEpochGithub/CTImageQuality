import random

import numpy as np
import torch
import torchvision
import tifffile
from PIL import Image
import LDCTIQAG2023_train as train_data
import json
from sklearn.utils import shuffle

import os.path as osp
import os

import analysis


def create_datasets(imgs_list, label_list, configs, final_train=False, patients_out=False, patient_ids_out=[3]):
    if final_train:
        return CT_Dataset(imgs_list, label_list, split="train", config=configs), None

    if patients_out:
        patient_ids = np.loadtxt(osp.join(osp.dirname(analysis.__file__), 'labels.txt'))
        patient_indices = [i for i, x in enumerate(patient_ids) if
                           x in patient_ids_out]  # np.where(patient_ids == patient_ids_out)[0]
        non_patient_indices = list(set(list(range(1000))) - set(patient_indices))
        print(len(patient_indices), len(non_patient_indices))
        train_dataset = CT_Dataset([imgs_list[x] for x in non_patient_indices],
                                   [label_list[x] for x in non_patient_indices], split="train",
                                   config=configs)
        test_dataset = CT_Dataset([imgs_list[x] for x in patient_indices], [label_list[x] for x in patient_indices],
                                  split="test", config=configs)
    else:
        # left_bound, right_bound = int(0.9 * len(imgs_list)), len(imgs_list)
        left_bound, right_bound = 900, 1000  # 0, 1000

        train_dataset = CT_Dataset(imgs_list[:left_bound] + imgs_list[right_bound:],
                                   label_list[:left_bound] + label_list[right_bound:], split="train", config=configs)
        test_dataset = CT_Dataset(imgs_list[left_bound:right_bound], label_list[left_bound:right_bound], split="test",
                                  config=configs)
    return train_dataset, test_dataset


def create_datalists(type="original"):
    if type == "mosaic":
        data_dir = osp.join(osp.dirname(train_data.__file__), "mosaic_dataset_25K", 'image')
        label_dir = osp.join(osp.dirname(train_data.__file__), "mosaic_dataset_25K", 'data.json')
    else:
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

def split_shuffle_image(image, num_parts=4):
    # Convert PIL Image to numpy array
    image_np = np.array(image)

    # Check that the image can be evenly divided into num_parts x num_parts
    # print(image_np.shape[0])
    # print(image_np.shape[1])
    assert image_np.shape[0] % num_parts == 0, "Image size must be evenly divisible by num_parts"
    assert image_np.shape[1] % num_parts == 0, "Image size must be evenly divisible by num_parts"

    # Split the image into patches
    patch_height = image_np.shape[0] // num_parts
    patch_width = image_np.shape[1] // num_parts
    patches = [image_np[i:i+patch_height, j:j+patch_width] for i in range(0, image_np.shape[0], patch_height) for j in range(0, image_np.shape[1], patch_width)]

    # Shuffle the patches
    patches = shuffle(patches)

    # Stitch the patches back together
    shuffled_image = np.block([[patches[num_parts * i + j] for j in range(num_parts)] for i in range(num_parts)])

    # Convert numpy array back to PIL Image
    shuffled_image = Image.fromarray(shuffled_image)

    return shuffled_image

def reverse_crop_image(image, crop_size):
    # Convert PIL Image to numpy array
    image_np = np.array(image)

    # Calculate the coordinates for the reverse crop
    start_x = (image_np.shape[1] - crop_size) // 2
    start_y = (image_np.shape[0] - crop_size) // 2

    # Perform reverse crop
    cropped_image = image_np[start_y:start_y+crop_size, start_x:start_x+crop_size]

    # Convert numpy array back to PIL Image
    cropped_image = Image.fromarray(cropped_image)

    return cropped_image

class CT_Dataset(torch.utils.data.Dataset):
    def __init__(self, imgs_list, label_list, split='validation', config={'img_size': 512}):
        self.imgs_list = imgs_list
        self.label_list = label_list
        self.split = split
        self.image_size = config['img_size']
        self.config = config
        self.crop_size = 460

        if self.split == 'train':

            operations = [torchvision.transforms.ToPILImage()]

            if self.config['Crop']:
                operations.append(torchvision.transforms.CenterCrop(self.crop_size))

            if self.config['ReverseCrop']:
                operations.append(torchvision.transforms.Lambda(lambda img: reverse_crop_image(img, self.crop_size)))

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
                # shear_degree = np.random.uniform(low=0.0, high=self.config['max_shear'])
                operations.append(torchvision.transforms.RandomApply([
                    torchvision.transforms.RandomAffine(degrees=self.config['max_shear'])
                ], p=0.1))
            split_shuffle_transform = torchvision.transforms.Lambda(lambda img: split_shuffle_image(img, num_parts=4))
            operations.append(split_shuffle_transform)
            operations += [torchvision.transforms.ToTensor()]

            if self.config['ShufflePatches']:
                operations.append(ShufflePatches(self.image_size // 4))

            self.transform = torchvision.transforms.Compose(operations)

        else:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.CenterCrop(self.crop_size),
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        x = self.imgs_list[idx]
        x = x.resize((self.image_size, self.image_size), Image.ANTIALIAS)
        x = np.array(x)
        # print("before  ", x.shape)
        x = self.transform(x)
        # print(self.transform)
        # print(x.shape)
        y = self.label_list[idx]

        return x, torch.tensor(y)


class ShufflePatches(object):
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, tensor):
        # Assuming tensor is of shape (C, H, W)
        _, height, width = tensor.size()
        patch_width = width // 4  # Divide the width into 4 equal patches
        patch_height = height // 4  # Divide the height into 4 equal patches

        # Create a list of patches
        patches = []
        for i in range(4):
            for j in range(4):
                left = j * patch_width
                upper = i * patch_height
                right = left + patch_width
                lower = upper + patch_height
                patch = tensor[:, upper:lower, left:right]
                patches.append(patch)

        # Shuffle the patches
        random.shuffle(patches)

        # Create a new tensor by concatenating the shuffled patches
        new_tensor = torch.cat(patches, dim=2)

        return new_tensor
