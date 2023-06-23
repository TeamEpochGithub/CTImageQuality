from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np

class CT_Dataset(Dataset):
    def __init__(self, lists, mode, norm, transform=None):
        self.lists = lists
        self.norm = norm
        self.mode = mode
        self.transform = transform
        self.target_ = []
        for i in range(len(self.lists)):
            self.target_.append(self.lists[i][1])

    def __len__(self):
        return len(self.lists)

    def __getitem__(self, idx):
        if self.mode == "denoise_task_2K" or self.mode=="AAPM":
            input_img, target_img = self.lists[idx]
            input_img, target_img = np.float32(np.load(input_img)), np.float32(np.load(target_img))
            if self.norm:
                input_img = (input_img - np.min(input_img)) / (np.max(input_img) - np.min(input_img))
                target_img = (target_img - np.min(target_img)) / (np.max(target_img) - np.min(target_img))
            augmentations = self.transform(image=input_img, mask=target_img)
            image = augmentations["image"]
            label = augmentations["mask"]
        else:
            input_img, label = self.lists[idx]
            input_img = np.float32(np.load(input_img))
            augmentations = self.transform(image=input_img)
            image = augmentations["image"]
            label = torch.tensor(label)
        return image, label