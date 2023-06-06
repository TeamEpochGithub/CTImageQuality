import numpy as np
import torch
import torchvision
from PIL import Image


class CT_Dataset(torch.utils.data.Dataset):
    def __init__(self, imgs_list, label_list, split, config):
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
