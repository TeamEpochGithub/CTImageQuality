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

    wandb.log({"avg_best_score": scores_dict['best_score'], "avg_best_score_epoch": scores_dict['best_score_epoch'], "avg_best_loss": scores_dict['avg_best_loss']})

    wandb.finish()


if __name__ == '__main__':
    # wandb.login()

    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'best_score',
            'goal': 'maximize'
        },
        'parameters': {
            "img_size": {
                'values': [256, 512]
            },
            'model': {
                'values': ['Efficientnet_Swin', 'Efficientnet_Swinv2', 'Resnet34_Swin', 'Resnet34_Swinv2']
            },
            'epochs': {
                'values': [250]
            },
            'batch_size': {
                'values': [2, 4, 8, 16]
            },
            'weight_decay': {
                # 'values': [0.0005, 0.005, 0.05]
                "distribution": "uniform",
                "min": 5e-5,
                "max": 1e-3
            },
            'lr': {
                # 'values': [1e-2, 1e-3, 3e-3, 2e-4, 3e-4, 1e-4, 5e-5]
                "distribution": "uniform",
                "min": 1e-4,
                "max": 1e-2
            },
            'min_lr': {
                # 'values': [1e-5, 1e-6, 1e-7, 1e-8]
                "distribution": "uniform",
                "min": 1e-8,
                "max": 1e-5
            },
            'RandomHorizontalFlip': {
                'values': [True, False]
            },
            'RandomVerticalFlip': {
                'values': [True, False]
            },
            'RandomRotation': {
                'values': [True, False]
            },
            'ZoomIn': {
                'values': [True, False]
            },
            'ZoomOut': {
                'values': [True, False]
            },
            'use_mix': {
                'values': [True, False]
            },
            'use_avg': {
                'values': [True, False]
            },
            'pretrain': {
                'values': [True, False]
            },
            'rotation_angle': {
                "distribution": "uniform",
                "min": 5,
                "max": 20
            },
            'zoomin_factor': {
                "distribution": "uniform",
                "min": 0.8,
                "max": 0.95
            },
            'zoomout_factor': {
                "distribution": "uniform",
                "min": 0.05,
                "max": 0.3
            }
        }
    }

    hypertune()

    # sweep_config = wandb.sweep("sweep.yaml", project="CTImageQuality-regression")
    # sweep_id = wandb.sweep(sweep_config, project="CTImageQuality-regression")
    # print(sweep_id)
    #
    # wandb.agent(sweep_id=sweep_id, project="CTImageQuality-regression", function=hypertune, count=7)
    # wandb.finish()
