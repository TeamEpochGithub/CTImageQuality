import os.path as osp
import numpy as np
import analysis
from datasets import create_datalists, CT_Dataset
from models.get_models import get_model
from train import train


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

    k_fold_patients_train(configs, wandb_single_experiment=False)
