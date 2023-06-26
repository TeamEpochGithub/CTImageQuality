import os
import wandb
from datasets import create_datalists, create_datasets
from train_local import train_local

os.environ['WANDB_API_KEY'] = ''


def train(training_data, parameters, context):
    # wandb.login()
    #
    # sweep_id = 'txs1e2qn'
    # wandb.agent(sweep_id, entity='epoch-iii', project='CTImageQuality')
    configs = {
        'pretrain': 'denoise',
        'img_size': 512,
        'model': 'DNCNN',
        'epochs': 180,
        'batch_size': 16,
        'weight_decay': 0.0003548,
        'lr': 0.003215,
        'min_lr': 0.000006463,
        'RandomHorizontalFlip': True,
        'RandomVerticalFlip': False,
        'RandomRotation': True,
        'ZoomIn': False,
        'ZoomOut': False,
        'use_mix': True,
        'use_avg': True,
        'XShift': False,
        'YShift': True,
        'RandomShear': True,
        'max_shear': 20,  # value in degrees
        'max_shift': 0.2,
        'rotation_angle': 20,
        'zoomin_factor': 0.95,
        'zoomout_factor': 0.05,
    }

    imgs_list, label_list = create_datalists(type="mosaic")  # type mosaic

    final_train = True

    train_dataset, test_dataset = create_datasets(imgs_list, label_list, configs, final_train=final_train,
                                                  patients_out=False, patient_ids_out=[0])
    # train_dataset, test_dataset = create_datasets(imgs_list, label_list, configs, final_train=final_train, patients_out=True, patient_ids_out=[3]])
    train_local(configs, train_dataset, test_dataset, wandb_single_experiment=False, final_train=final_train)


if __name__ == '__main__':
    train(None, None, None)
