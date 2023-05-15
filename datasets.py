import numpy as np
import torch
import torchvision
from PIL import Image


class CT_Dataset(torch.utils.data.Dataset):
    def __init__(self, imgs_list, label_list, split, config):
        self.imgs_list = imgs_list
        self.label_list = label_list
        self.split = split
        self.image_size = config['image_size']
        self.config = config

        if self.split == 'train':

            operations = [torchvision.transforms.ToPILImage()]

            if self.config['augment']['RandomHorizontalFlip']:
                operations.append(torchvision.transforms.RandomHorizontalFlip())

            if self.config['augment']['RandomVerticalFlip']:
                operations.append(torchvision.transforms.RandomVerticalFlip())

            if self.config['augment']['RandomRotation']:
                operations.append(torchvision.transforms.RandomRotation(15))

            if self.config['augment']['ZoomIn']:
                operations.append(torchvision.transforms.RandomApply([
                    torchvision.transforms.CenterCrop(size=480),
                    torchvision.transforms.Resize(size=(self.image_size, self.image_size)),
                ], p=0.1))
            if self.config['augment']['ZoomOut']:
                operations.append(torchvision.transforms.RandomApply([
                    torchvision.transforms.Pad(padding=20),
                    torchvision.transforms.Resize((self.image_size, self.image_size)),
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
