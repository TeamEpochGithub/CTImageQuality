import os
import os.path as osp
import random
import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pytorch_warmup as warmup
from scipy.stats import pearsonr, spearmanr, kendalltau
import wandb
from datasets import create_datalists, create_datasets
from models.get_models import get_model


def set_seed(seed):
    """Set all random seeds and settings for reproducibility (deterministic behavior)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(0)


def valid(model, test_dataset, best_score, best_score_epoch, epoch, wandb_single_experiment=False):
    model.eval()
    total_pred = []
    total_gt = []
    aggregate_results = dict()

    t = tqdm(enumerate(test_dataset), total=len(test_dataset), desc="validation", colour='blue')
    with torch.no_grad():
        for i, (img, label) in t:
            img = img.unsqueeze(0).float()
            pred = model(img.cuda())
            pred_new = pred.cpu().numpy().squeeze()
            label_new = label.cpu().numpy()
            # print(round(pred_new[0], 2), label_new)
            total_pred.append(pred_new)
            total_gt.append(label_new)
            if i == len(test_dataset) - 1:
                # errors = [abs(x - float(y)) for x, y in zip(total_pred, total_gt)]
                total_pred = np.array(total_pred)
                total_gt = np.array(total_gt)
                aggregate_results["plcc"] = abs(pearsonr(total_pred, total_gt)[0])
                aggregate_results["srocc"] = abs(spearmanr(total_pred, total_gt)[0])
                aggregate_results["krocc"] = abs(kendalltau(total_pred, total_gt)[0])
                aggregate_results["overall"] = abs(pearsonr(total_pred, total_gt)[0]) + abs(
                    spearmanr(total_pred, total_gt)[0]) + abs(kendalltau(total_pred, total_gt)[0])
                std = np.std(total_pred - total_gt)
                aggregate_results["std"] = std
                mean = np.mean(np.abs(total_pred - total_gt))
                aggregate_results["mean"] = mean
                t.set_postfix({key: round(value, 3) for key, value in aggregate_results.items()})
    # import matplotlib.pyplot as plt
    # plt.hist(errors, bins=20)
    # plt.show()

    aggregate_results['epoch'] = epoch
    if aggregate_results["overall"] > best_score:
        # print("new best model saved")
        best_score = aggregate_results["overall"]
        best_score_epoch = epoch

        if not os.path.exists('output'):
            os.makedirs('output')
        torch.save(model.state_dict(), osp.join('output', "model.pth"))
        if wandb_single_experiment:
            wandb.save("model.pth")

    return best_score, best_score_epoch


def train_local(configs, train_dataset, test_dataset, wandb_single_experiment=False, final_train=False):
    model = get_model(configs)
    if 'Swin' in configs['model']:
        model = model(configs=configs)
    model = model.cuda()

    if configs['pretrain'] != 'None':
        file_dict = {'discrete_classification': "pretrain_weight_classification.pkl",
                     'denoise': "pretrain_weight_denoise.pkl"}
        weight_path = osp.join(osp.dirname(osp.abspath(__file__)), "pretrained_weights", configs['model'],
                               file_dict[configs['pretrain']])

        if os.path.exists(weight_path):
            pre_weights = torch.load(weight_path, map_location=torch.device("cuda"))
            for name, param in model.named_parameters():
                if name in pre_weights:
                    param.data.copy_(pre_weights[name])

    optimizer = optim.AdamW(model.parameters(), lr=configs['lr'], betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=configs['weight_decay'])
    train_loader = DataLoader(train_dataset, batch_size=configs['batch_size'], shuffle=True)
    num_steps = len(train_loader) * configs['epochs']
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=configs['min_lr'])
    warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)

    best_score = 0
    best_loss = 10
    best_score_epoch = 0
    for epoch in range(configs['epochs']):
        losses = 0
        model.train()

        t = tqdm(enumerate(train_loader), total=len(train_loader), desc="epoch " + f"{epoch:04d}", colour='cyan')

        for i, (image, target) in t:
            image = image.cuda()
            target = target.cuda()
            pred = model(image)
            loss = F.mse_loss(pred.squeeze(), target)
            losses += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with warmup_scheduler.dampening():
                lr_scheduler.step()

            if i == len(train_loader) - 1:
                t.set_postfix(
                    {"loss": round(float(losses / len(train_dataset)), 5),
                     "lr": round(lr_scheduler.get_last_lr()[0], 8)})

        loss = float(losses / len(train_dataset))
        if loss < best_loss:
            best_loss = loss

        if epoch % 1 == 0 and not final_train:
            best_score, best_score_epoch = valid(model, test_dataset, best_score, best_score_epoch, epoch,
                                                 wandb_single_experiment)

        if (epoch + 1) % configs['epochs'] == 0:
            if not os.path.exists('output'):
                os.makedirs('output')
            if final_train:
                torch.save(model.state_dict(), osp.join('output', f"{configs['model']}_epoch_{epoch}_alldata.pth"))
            else:
                torch.save(model.state_dict(), osp.join('output', f"{configs['model']}_epoch_{epoch}_9010.pth"))
            print("Model saved!")

    return model.state_dict()  # {"best_score": best_score, "best_score_epoch": best_score_epoch, "best_loss": best_loss}


if __name__ == '__main__':
    configs = {
        'pretrain': 'denoise',
        'img_size': 512,
        'model': 'EDCNN3',
        'epochs': 50,
        'batch_size': 16,
        'weight_decay': 1e-3,
        'lr': 3e-4,
        'min_lr': 1e-6,
        'ShufflePatches': False,
        'RandomHorizontalFlip': True,
        'RandomVerticalFlip': False,
        'RandomRotation': True,
        'ZoomIn': False,
        'ZoomOut': False,
        'use_mix': True,
        'use_avg': True,
        'XShift': True,
        'YShift': True,
        'RandomShear': False,
        'max_shear': 20,  # value in degrees
        'max_shift': 0.05,
        'rotation_angle': 3,
        'zoomin_factor': 0.95,
        'zoomout_factor': 0.05,
    }

    imgs_list, label_list = create_datalists(type="original")  # type mosaic

    torch.cuda.set_device(1)

    final_train = False

    train_dataset, test_dataset = create_datasets(imgs_list, label_list, configs, final_train=final_train,
                                                  patients_out=False, patient_ids_out=[3])
    # train_dataset, test_dataset = create_datasets(imgs_list, label_list, configs, final_train=final_train, patients_out=True, patient_ids_out=[3]])
    train_local(configs, train_dataset, test_dataset, wandb_single_experiment=False, final_train=final_train)
