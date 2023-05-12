import numpy as np
import torch
import torchvision
from PIL import Image

class CT_Dataset(torch.utils.data.Dataset):
    def __init__(self, imgs_list, label_list, split):
        self.imgs_list = imgs_list
        self.label_list = label_list
        self.split = split

        if self.split == 'train':
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.RandomRotation(15),
                # torchvision.transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.7, 0.9)),
                # torchvision.transforms.RandomPerspective(),
                torchvision.transforms.ToTensor(),
            ])
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
        x = x.resize((256, 256), Image.ANTIALIAS)
        x = np.array(x)
        x = self.transform(x)
        y = self.label_list[idx]

        return x, torch.tensor(y)
