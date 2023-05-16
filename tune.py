from datasets import CT_Dataset
from evaluate import create_datalists
from models.efficient_swin import Efficientnet_Swin
from models.efficient_swinv2 import Efficientnet_Swinv2
from models.res34_swin import Resnet34_Swin
from models.res34_swinv2 import Resnet34_Swinv2
from train import train
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
        project=f"CTImageQuality-regression",
        notes="My first experiment",
        tags=["baselines"],
        config=config_defaults,
    )
    print("config:", wandb.config)

    models = {'Efficientnet_Swin': Efficientnet_Swin, 'Efficientnet_Swinv2': Efficientnet_Swinv2,
              'Resnet34_Swin': Resnet34_Swin, 'Resnet34_Swinv2': Resnet34_Swinv2}

    model = models[wandb.config.model]

    imgs_list, label_list = create_datalists()

    left_bound, right_bound = 900, 1000

    train_dataset = CT_Dataset(imgs_list[:left_bound] + imgs_list[right_bound:],
                               label_list[:left_bound] + label_list[right_bound:], split="train", config=wandb.config)
    test_dataset = CT_Dataset(imgs_list[left_bound:right_bound], label_list[left_bound:right_bound], split="test",
                              config=wandb.config)

    scores_dict = train(model, wandb.config, train_dataset, test_dataset)

    wandb.log({"best_score": scores_dict['best_score']})


if __name__ == '__main__':
    wandb.login()

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
                'values': [3]
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

    # sweep_id = wandb.sweep(sweep_config, project="CTImageQuality-regression")

    wandb.agent(sweep_id='4nf8mksa', project="CTImageQuality-regression", function=hypertune, count=7)
    wandb.finish()
