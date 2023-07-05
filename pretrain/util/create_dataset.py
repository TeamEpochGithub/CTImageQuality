import os
import random
import json
import albumentations as A
from glob import glob

import numpy as np
from albumentations.pytorch import ToTensorV2
import os.path as osp

from torchvision.transforms import ToTensor

from pretrain.pretrain_dataloaders.classic_dataset import CT_Dataset
import pretrain
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import crop
import torch


def create_datasets(parameters):
    pretrain_path = osp.dirname(pretrain.__file__)
    folder = parameters["folder"]
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        ToTensorV2()
    ])

    test_transform = A.Compose([
        ToTensorV2()
    ])

    data_path = osp.join(pretrain_path, 'pretrain_data', folder)
    lists = []
    if folder == "denoise_task_2K":
        input_path = sorted(glob(os.path.join(data_path, '*input*.npy')))
        target_path = sorted(glob(os.path.join(data_path, '*target*.npy')))
        for i in range(len(input_path)):
            lists.append((input_path[i], target_path[i]))
    elif folder == "AAPM":
        train_FD_path = sorted(glob(os.path.join(data_path, "train_set", 'FD_NPY', '*FD*.npy')))
        train_QD_path = sorted(glob(os.path.join(data_path, "train_set", 'QD_NPY', '*QD*.npy')))
        test_FD_path = sorted(glob(os.path.join(data_path, "test_set", 'FD_NPY', '*FD*.npy')))
        test_QD_path = sorted(glob(os.path.join(data_path, "test_set", 'QD_NPY', '*QD*.npy')))
    else:
        labels = osp.join(data_path, 'label.json')
        with open(labels, 'r') as f:
            label_dict = json.load(f)
        for key in label_dict:
            lists.append((osp.join(data_path, "image", key), label_dict[key]))

    train_lists = []
    test_lists = []
    if folder == "AAPM":
        aapm_train = []
        aapm_label = []
        for i in range(len(train_FD_path)):
            aapm_train.append(train_QD_path[i])
            aapm_label.append(train_FD_path[i])
        for i in range(len(test_FD_path)):
            test_lists.append((test_QD_path[i], test_FD_path[i]))
        # train_dataset = CT_Dataset(train_lists, transform=train_transform, norm=False, mode=folder)
        train_dataset = CustomAAPMDataset(aapm_train, aapm_label)
        test_dataset = CT_Dataset(test_lists, transform=test_transform, norm=False, mode=folder)
    else:
        random.shuffle(lists)
        train_lists = lists[:int(len(lists) * parameters["split_ratio"])]
        test_lists = lists[int(len(lists) * parameters["split_ratio"]):]
        train_dataset = CT_Dataset(train_lists, transform=train_transform, norm=True, mode=folder)
        test_dataset = CT_Dataset(test_lists, transform=test_transform, norm=True, mode=folder)
    return train_dataset, test_dataset


class CustomAAPMDataset(Dataset):
    def __init__(self, images, target_images, crop_size=(64, 64)):
        self.images = images
        self.target_images = target_images
        self.crop_size = crop_size
        self.to_tensor = ToTensor()

    def __getitem__(self, index):
        image = torch.from_numpy(np.load(self.images[index]))
        target_image = torch.from_numpy(np.load(self.target_images[index]))

        # print(self.images[index])
        # print(np.load(self.images[index]))
        # print(type(image))
        # Generate random top-left coordinates for cropping
        image_height, image_width = image.shape
        top = torch.randint(0, image_height - self.crop_size[0] + 1, (1,))
        left = torch.randint(0, image_width - self.crop_size[1] + 1, (1,))

        # Apply the same random crop to both the image and target_image
        cropped_image = crop(image, top.item(), left.item(), self.crop_size[0], self.crop_size[1])
        cropped_target_image = crop(target_image, top.item(), left.item(), self.crop_size[0], self.crop_size[1])

        # # Convert the cropped images to tensors
        # image_tensor = self.to_tensor(cropped_image)
        # target_image_tensor = self.to_tensor(cropped_target_image)
        cropped_image, cropped_target_image = cropped_image.unsqueeze(0), cropped_target_image.unsqueeze(0)
        assert cropped_image.shape == cropped_target_image.shape
        return cropped_image, cropped_target_image

    def __len__(self):
        return len(self.images)
