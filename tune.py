import torch.cuda

from k_fold import k_fold_patients_train
from models.efficient_swin import Efficientnet_Swin
import wandb


def hypertune():
    config_defaults = {
        'model': Efficientnet_Swin,
        'epochs': 251,
        'batch_size': 8,
        'weight_decay': 0.0005,
        'lr': 3e-4,
        'min_lr': 1e-6,
        'RandomHorizontalFlip': False,
        'RandomVerticalFlip': False,
        'RandomRotation': False,
        'ZoomIn': False,
        'ZoomOut': False,
        'use_mix': True,
        'use_avg': True,
        'rotation_angle': 15,
        'zoomout_factor': 0.15,
        'zoomin_factor': 0.9
    }

    wandb.init(
        # project=f"CTImageQuality-regression",
        # notes="My first experiment",
        # tags=["baselines"],
        # config=config_defaults,
    )
    print("config:", wandb.config)

    scores_dict = k_fold_patients_train(wandb.config, wandb_single_experiment=False)

    wandb.log({"avg_best_score": scores_dict['avg_best_score'], "avg_best_score_epoch": scores_dict['avg_best_score_epoch'], "avg_best_loss": scores_dict['avg_best_loss']})

    wandb.finish()


if __name__ == '__main__':
    hypertune()
