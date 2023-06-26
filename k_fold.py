import os.path as osp
import numpy as np
import torch.cuda

import analysis
from datasets import create_datalists, CT_Dataset
from models.get_models import get_model
from train_local import train


def k_fold_patients_train(configs, wandb_single_experiment=False):
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

        scores_dict = train(configs, train_dataset, test_dataset, wandb_single_experiment, final_train=False)
        best_scores.append(scores_dict['best_score'])
        best_score_epochs.append(scores_dict['best_score_epoch'])
        best_losses.append(scores_dict['best_loss'])

    print({'avg_best_score': np.mean(best_scores), 'avg_best_score_epoch': np.mean(best_score_epochs), 'avg_best_loss': np.mean(best_losses)})
    return {'avg_best_score': np.mean(best_scores), 'avg_best_score_epoch': np.mean(best_score_epochs), 'avg_best_loss': np.mean(best_losses)}


if __name__ == '__main__':
    configs = {
        'pretrain': 'denoise',
        'img_size': 512,
        'model': 'Efficientnet_B1',
        'epochs': 25,
        'batch_size': 16,
        'weight_decay': 0.000494,
        'lr': 0.009666,
        'min_lr': 0.000006463,
        'RandomHorizontalFlip': True,
        'RandomVerticalFlip': True,
        'RandomRotation': True,
        'ZoomIn': False,
        'ZoomOut': True,
        'use_mix': True,
        'use_avg': False,
        'rotation_angle': 11.168,
        'zoomin_factor': 0.8033,
        'zoomout_factor': 0.1014,
    }
    # torch.cuda.set_device(1)
    k_fold_patients_train(configs, wandb_single_experiment=False)
