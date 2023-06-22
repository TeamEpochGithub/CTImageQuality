import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error


# Your existing dataloader code here...
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

# Function to train GMM
from torchvision import models

# Define a pretrained ResNet50 model
class ResNet50Features(models.ResNet):
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Extracted features

        return x

def train_feature_extractor(dataloader):
    # Instantiate the model
    model = ResNet50Features(models.resnet.Bottleneck, [3, 4, 6, 3])
    model.eval()  # Set the model to evaluation mode

    all_images = []

    for batch in dataloader:
        images, labels = batch
        with torch.no_grad():
            # Extract features
            features = model(images).numpy()
        all_images.append(features)

    all_images = np.concatenate(all_images)

    # Train GMM on the extracted features
    gmm = GaussianMixture(n_components=5, covariance_type='diag')
    gmm.fit(all_images)

    return gmm

def evaluate_feature_extractor(gmm, dataloader):
    all_labels = []
    all_preds = []

    for batch in dataloader:
        images, labels = batch
        with torch.no_grad():
            # Extract features
            features = model(images).numpy()

        # Predict with GMM
        preds = gmm.predict(features)

        all_labels.append(labels.numpy())
        all_preds.append(preds)

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    mse = mean_squared_error(all_labels, all_preds)
    print(f'Mean Squared Error: {mse}')

# Replace the training and evaluation function calls
# Train GMM
gmm = train_feature_extractor(train_dataloader)

# Evaluate GMM
evaluate_feature_extractor(gmm, test_dataloader)


if __name__ == '__main__':
    configs = {
        'pretrain': None,
        'img_size': 512,
        'model': 'Resnet50',
        'epochs': 150,
        'batch_size': 16,
        'weight_decay': 3e-4,
        'lr': 6e-3,
        'min_lr': 5e-6,
        'RandomHorizontalFlip': True,
        'RandomVerticalFlip': True,
        'RandomRotation': True,
        'ZoomIn': True,
        'ZoomOut': False,
        'use_mix': False,
        'use_avg': False,
        'XShift': False,
        'YShift': False,
        'RandomShear': False,
        'max_shear': 30,  # value in degrees
        'max_shift': 0.5,
        'rotation_angle': 12.4,
        'zoomin_factor': 0.9,
        'zoomout_factor': 0.27,
    }

    # Create datasets
    train_dataset, test_dataset = create_datasets(*create_datalists(), configs=configs)

    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

    # Train GMM
    gmm = train_gmm(train_dataloader)

    # Evaluate GMM
    evaluate_gmm(gmm, test_dataloader)
