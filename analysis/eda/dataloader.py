import json

import tifffile
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Lambda, Resize, ToPILImage, RandomRotation, RandomVerticalFlip, \
    RandomHorizontalFlip, RandomResizedCrop, RandomApply
import matplotlib.pyplot as plt
import os
import pandas as pd
from torchvision.io import read_image
import numpy as np
import wandb

class CTIDataset(Dataset):
    def __init__(self, img_dir, labels, file_names, split="train"):

        self.img_dir = img_dir
        self.labels = labels
        self.file_names = file_names
        self.split = split

        if self.split == 'train':
            self.transform = Compose([
                ToPILImage(),
                # RandomHorizontalFlip(),
                # RandomVerticalFlip(),
                RandomRotation(15),
                ToTensor(),
            ])

            # self.transform = Compose([
            #     ToPILImage(),
            #     # Random rotation within the range of -15 to 15 degrees
            #     RandomRotation(degrees=(-15, 15)),
            #     # Random resized crop
            #     RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0)),
            #     # Convert the image to tensor
            #     ToTensor(),])
        else:
            self.transform = Compose([
                ToPILImage(),
                ToTensor(),
                # torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        # self.target_transform = target_transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        img_path = os.path.join(self.img_dir, file_name)
        image = tifffile.imread(img_path)
        label = torch.tensor(self.labels[file_name], dtype=torch.float)
        if self.transform:
            image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        # print(image, label)
        return image, label


