import os
import pydicom
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
class AAPMDataset(Dataset):
    def __init__(self, input_dir, label_dir, transform=None, mode='train'):
        self.input_dir = input_dir
        self.label_dir = label_dir
        self.transform = transform
        self.samples = []

        split = 0.9
        for i in range(len(os.listdir(self.input_dir))):
            image_dir_path = os.path.join(self.input_dir, os.listdir(self.input_dir)[i])
            label_dir_path = os.path.join(self.label_dir, os.listdir(self.label_dir)[i])

            for j in range(len(os.listdir(image_dir_path))):
                patient_input_path = os.path.join(image_dir_path, os.listdir(image_dir_path)[j], os.listdir(self.input_dir)[i])
                # print(patient_input_path)
                patient_label_path = os.path.join(label_dir_path, os.listdir(label_dir_path)[j], os.listdir(self.label_dir)[i])

                # print(os.listdir(patient_input_path))

                for k in range(len(os.listdir(patient_input_path))):
                    # print(file)
                    input_file_path = os.path.join(patient_input_path, os.listdir(patient_input_path)[k])
                    label_file_path = os.path.join(patient_label_path, os.listdir(patient_label_path)[k])
                    # print(label_file_path)
                    if os.path.exists(label_file_path):
                        # print('file found')
                        self.samples.append((input_file_path, label_file_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_file_path, label_file_path = self.samples[idx]

        input_image = pydicom.dcmread(input_file_path).pixel_array
        input_image = torch.from_numpy(input_image.astype(np.float32)).unsqueeze(0)

        label_image = pydicom.dcmread(label_file_path).pixel_array
        label_image = torch.from_numpy(label_image.astype(np.float32)).unsqueeze(0)

        # Normalize the images manually using Min-Max normalization
        input_image = (input_image - torch.min(input_image)) / (torch.max(input_image) - torch.min(input_image))
        label_image = (label_image - torch.min(label_image)) / (torch.max(label_image) - torch.min(label_image))

        if self.transform:
            input_image = self.transform(input_image)
            label_image = self.transform(label_image)

        return input_image, label_image