import os.path as osp

import numpy as np

import wandb

import analysis
from datasets import create_datalists, CT_Dataset
from models.efficient_swin import Efficientnet_Swin
from models.efficient_swinv2 import Efficientnet_Swinv2
from models.efficientnet import load_efficientnet_model
from models.res34_swin import Resnet34_Swin
from models.res34_swinv2 import Resnet34_Swinv2
from models.resnet import load_resnet_model
from train import train


def k_fold_patients_train(model, configs, wandb_run=False):
    best_scores = []
    best_score_epochs = []
    best_losses = []

    imgs_list, label_list = create_datalists()
    patient_ids = np.loadtxt(osp.join(osp.dirname(analysis.__file__), 'labels.txt'))

    for i in range(4):
        patient_indices = np.where(patient_ids == i)[0]
        print(len(patient_indices))
        non_patient_indices = list(set(list(range(1000))) - set(patient_indices))
        train_dataset = CT_Dataset([imgs_list[x] for x in non_patient_indices],
                                   [label_list[x] for x in non_patient_indices], split="train",
                                   config=configs)
        test_dataset = CT_Dataset([imgs_list[x] for x in patient_indices], [label_list[x] for x in patient_indices],
                                  split="test", config=configs)

        scores_dict = train(model, configs, train_dataset, test_dataset, wandb_run)
        best_scores.append(scores_dict['best_score'])
        best_score_epochs.append(scores_dict['best_score_epoch'])
        best_losses.append(scores_dict['best_loss'])

    print({'avg_best_score': np.mean(best_scores), 'avg_best_score_epoch': np.mean(best_score_epochs), 'avg_best_loss': np.mean(best_losses)})

    if wandb_run:
        wandb.log({'avg_best_score': np.mean(best_scores), 'avg_best_score_epoch': np.mean(best_score_epochs),
                   'avg_best_loss': np.mean(best_losses)})
        wandb.finish()


if __name__ == '__main__':
    configs = {
        'pretrain': 'None',
        'img_size': 512,
        'model': 'Resnet50',
        'epochs': 10,
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
        'use_avg': True,
        'rotation_angle': 12.4,
        'zoomin_factor': 0.9,
        'zoomout_factor': 0.27,
    }

    models = {'Resnet18': load_resnet_model('18', configs['pretrain']),
              'Resnet50': load_resnet_model('50', configs['pretrain']),
              'Resnet152': load_resnet_model('152', configs['pretrain']),
              'Efficientnet_B0': load_efficientnet_model('b0', configs['pretrain']),
              'Efficientnet_B4': load_efficientnet_model('b4', configs['pretrain']),
              'Efficientnet_B7': load_efficientnet_model('b7', configs['pretrain']),
              'Efficientnet_Swin': Efficientnet_Swin, 'Efficientnet_Swinv2': Efficientnet_Swinv2,
              'Resnet34_Swin': Resnet34_Swin, 'Resnet34_Swinv2': Resnet34_Swinv2}

    model = models['Resnet50']

    k_fold_patients_train(model, configs, wandb_run=False)
