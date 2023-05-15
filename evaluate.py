from datasets import CT_Dataset
import tifffile
from PIL import Image
import LDCTIQAG2023_train as train_data
import json

import os.path as osp
import os

from models.efficient_swin import Efficientnet_Swin
from models.efficient_swinv2 import Efficientnet_Swinv2
from models.res34_swin import Resnet34_Swin
from models.res34_swinv2 import Resnet34_Swinv2
from train import train
import wandb
import numpy as np


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


def evaluate_k_fold(config, name="model", folds=5):
    wandb.login()
    run = wandb.init(
        project=f"CTImageQuality-regression",
        notes="My first experiment",
        tags=["baselines"],
        config=config,
        name=f"{name}-average-{folds}fold"
    )

    imgs_list, label_list = create_datalists()

    best_scores = []
    best_score_epochs = []
    best_losses = []
    for i in range(folds):
        left_bound, right_bound = i * int(len(imgs_list) / folds), (i + 1) * int(len(imgs_list) / folds)

        train_dataset = CT_Dataset(imgs_list[:left_bound] + imgs_list[right_bound:], label_list[:left_bound] + label_list[right_bound:], split="train")
        test_dataset = CT_Dataset(imgs_list[left_bound:right_bound], label_list[left_bound:right_bound], split="test")

        scores_dict = train(config["model"], config, train_dataset, test_dataset)

        best_scores.append(scores_dict['best_score'])
        best_score_epochs.append(scores_dict['best_score_epoch'])
        best_losses.append(scores_dict['best_loss'])

    wandb.log({'avg_best_score': np.mean(best_scores), 'avg_best_score_epoch': np.mean(best_score_epochs), 'avg_best_loss': np.mean(best_losses)})
    wandb.finish()


if __name__ == '__main__':

    names = ['Efficientnet_Swin', 'Efficientnet_Swinv2', 'Resnet34_Swin', 'Resnet34_Swinv2']
    models = [Efficientnet_Swin, Efficientnet_Swinv2, Resnet34_Swin, Resnet34_Swinv2]

    efficient_swin_config = {
        "model": Efficientnet_Swin,
        "batch_size": 8,
        "epochs": 151,
        "lr": 3e-4,
        "min_lr": 1e-6,
        "weight_decay": 1e-4,
        "name": 'Efficientnet_Swin'
    }

    efficient_swinv2_config = {
        "model": Efficientnet_Swinv2,
        "batch_size": 8,
        "epochs": 151,
        "lr": 3e-4,
        "min_lr": 1e-6,
        "weight_decay": 1e-4,
        "name": 'Efficientnet_Swinv2'
    }

    resnet_swin_config = {
        "model": Resnet34_Swin,
        "batch_size": 16,
        "epochs": 151,
        "lr": 3e-4,
        "min_lr": 1e-6,
        "weight_decay": 1e-4,
        "name": 'Resnet34_Swin'
    }

    resnet_swinv2_config = {
        "model": Resnet34_Swinv2,
        "batch_size": 16,
        "epochs": 151,
        "lr": 3e-4,
        "min_lr": 1e-6,
        "weight_decay": 1e-4,
        "name": 'Resnet34_Swinv2'
    }

    all_configs = [efficient_swin_config, efficient_swinv2_config, resnet_swin_config, resnet_swinv2_config]

    for config in all_configs:
        evaluate_k_fold(config, name=config['name'], folds=5)
