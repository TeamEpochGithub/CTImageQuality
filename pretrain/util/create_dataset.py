import os
import random
import json
import albumentations as A
from glob import glob
from albumentations.pytorch import ToTensorV2
import os.path as osp
from pretrain.pretrain_dataloaders.classic_dataset import CT_Dataset
from pretrain.pretrain_dataloaders.aapm_dataset import AAPMDataset
import pretrain
from torch.utils.data import random_split


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
        for i in range(len(train_FD_path)):
            train_lists.append((train_QD_path[i], train_FD_path[i]))
        for i in range(len(test_FD_path)):
            test_lists.append((test_QD_path[i], test_FD_path[i]))
        train_dataset = CT_Dataset(train_lists, transform=train_transform, norm=False, mode=folder)
        test_dataset = CT_Dataset(test_lists, transform=test_transform, norm=False, mode=folder)
    else:
        random.shuffle(lists)
        train_lists = lists[:int(len(lists) * parameters["split_ratio"])]
        test_lists = lists[int(len(lists) * parameters["split_ratio"]):]
        train_dataset = CT_Dataset(train_lists, transform=train_transform, norm=True, mode=folder)
        test_dataset = CT_Dataset(test_lists, transform=test_transform, norm=True, mode=folder)
    return train_dataset, test_dataset
