import numpy as np
import torch
import torchvision
import tifffile
from PIL import Image
import LDCTIQAG2023_train as train_data
import json

import os.path as osp
import os


def create_datasets(imgs_list, label_list, configs):
    one_patient_out = True
    if one_patient_out:
        patient_indices = [0, 1, 2, 3, 5, 8, 9, 12, 20, 24, 29, 33, 36, 37, 40, 41, 42, 53, 58, 64, 67, 74, 79, 88, 90,
                           93, 97, 108, 109, 111, 112, 113, 117, 120, 126, 127, 128, 131, 132, 133, 134, 137, 143, 148,
                           150, 153, 155, 166, 168, 172, 184, 186, 187, 191, 194, 208, 210, 211, 220, 223, 227, 228,
                           231, 234, 241, 249, 252, 256, 257, 259, 264, 267, 269, 272, 277, 278, 281, 284, 286, 287,
                           294, 296, 299, 300, 304, 306, 308, 312, 314, 315, 318, 321, 326, 327, 328, 329, 341, 343,
                           349, 363, 365, 367, 369, 370, 372, 376, 380, 391, 397, 404, 405, 407, 410, 412, 425, 427,
                           429, 440, 446, 448, 449, 455, 457, 464, 465, 469, 481, 489, 491, 498, 509, 517, 518, 520,
                           527, 530, 533, 538, 540, 546, 547, 553, 564, 565, 568, 569, 574, 582, 589, 591, 600, 604,
                           607, 608, 609, 610, 612, 621, 622, 637, 638, 643, 651, 652, 653, 659, 660, 665, 668, 669,
                           670, 671, 674, 677, 685, 691, 694, 695, 699, 701, 714, 725, 728, 729, 740, 741, 742, 743,
                           746, 748, 752, 754, 759, 761, 764, 766, 767, 773, 774, 775, 776, 777, 783, 789, 790, 791,
                           792, 795, 796, 798, 799, 805, 806, 818, 825, 831, 833, 837, 838, 847, 849, 852, 855, 859,
                           863, 866, 868, 877, 879, 888, 892, 893, 894, 896, 904, 907, 910, 914, 923, 927, 942, 943,
                           945, 952, 965, 967, 980, 986, 993, 996]
        print(len(patient_indices))
        non_patient_indices = list(set(list(range(1000))) - set(patient_indices))
        train_dataset = CT_Dataset([imgs_list[x] for x in non_patient_indices],
                                   [label_list[x] for x in non_patient_indices], split="train",
                                   config=configs)
        test_dataset = CT_Dataset([imgs_list[x] for x in patient_indices], [label_list[x] for x in patient_indices],
                                  split="test", config=configs)
    else:
        left_bound, right_bound = 900, 1000

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
                operations.append(torchvision.transforms.RandomAffine(degrees=0, translate=shifts))

            if self.config['RandomShear']:
                shear_degree = np.random.uniform(low=0.0, high=self.config['max_shear'])
                operations.append(torchvision.transforms.RandomAffine(degrees=0, shear=shear_degree))

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
