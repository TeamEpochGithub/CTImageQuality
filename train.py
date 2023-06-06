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
from datasets import CT_Dataset, create_datalists
from models.efficient_swin import Efficientnet_Swin
from models.efficient_swinv2 import Efficientnet_Swinv2
from models.efficientnet import load_efficientnet_model
from models.res34_swin import Resnet34_Swin
from models.res34_swinv2 import Resnet34_Swinv2
from models.resnet import load_resnet_model


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


def valid(model, test_dataset, best_score, best_score_epoch, epoch, wandb_run=False):
    model.eval()
    total_pred = []
    total_gt = []
    aggregate_results = dict()

    t = tqdm(enumerate(test_dataset), total=len(test_dataset), desc="validation", colour='blue')
    with torch.no_grad():
        for i, (img, label) in t:
            img = img.unsqueeze(0).float()
            pred = model(img.cuda())
            pred_new = pred.cpu().numpy().squeeze(0)
            label_new = label.cpu().numpy()
            # print(round(pred_new[0], 2), label_new)
            total_pred.append(pred_new[0])
            total_gt.append(label_new)
            if i == len(test_dataset) - 1:
                errors = [abs(x - float(y)) for x, y in zip(total_pred, total_gt)]
                for ind, er in enumerate(errors):
                    if er < 0.15:
                        patient_indices = [0, 1, 3, 8, 20, 24, 29, 33, 36, 41, 42, 58, 79, 88, 90, 93, 97, 109, 112,
                                           128, 132, 133, 134,
                                           137, 155, 168, 187, 191, 208, 210, 220, 227, 228, 231, 234, 241, 249, 252,
                                           256, 257, 259,
                                           264, 267,
                                           277, 284, 286, 287, 294, 299, 308, 314, 318, 321, 326, 328, 341, 349, 363,
                                           365, 370, 372,
                                           380, 397, 404, 405, 407, 412,
                                           429, 440, 446, 448, 455, 457, 464, 465, 469, 481, 489, 491, 498,
                                           509, 517, 518, 520, 527, 530, 533, 538, 546, 547, 553, 564, 565, 568, 569,
                                           589,
                                           591, 608, 609, 610, 612, 621, 638, 643, 653, 668, 669, 671, 674, 677, 685,
                                           691,
                                           699, 701, 714, 728, 729, 740, 742, 743, 746, 754, 759, 761, 764, 766, 767,
                                           790,
                                           791, 796, 818, 825, 837, 855, 859, 866, 868, 877, 888, 892, 904, 907, 910,
                                           914,
                                           942, 943, 945, 965, 980, 996
                                           ]
                        print(patient_indices[ind])
                print([f"{round(x, 2):.2f}" for x in total_pred])
                print([f"{float(x):.2f}" for x in total_gt])
                total_pred = np.array(total_pred)
                total_gt = np.array(total_gt)
                aggregate_results["plcc"] = abs(pearsonr(total_pred, total_gt)[0])
                aggregate_results["srocc"] = abs(spearmanr(total_pred, total_gt)[0])
                aggregate_results["krocc"] = abs(kendalltau(total_pred, total_gt)[0])
                aggregate_results["overall"] = abs(pearsonr(total_pred, total_gt)[0]) + abs(
                    spearmanr(total_pred, total_gt)[0]) + abs(kendalltau(total_pred, total_gt)[0])
                std = np.std(total_pred - total_gt)
                aggregate_results["std"] = std
                t.set_postfix({key: round(value, 3) for key, value in aggregate_results.items()})
    import matplotlib.pyplot as plt
    plt.hist(errors, bins=20)
    plt.show()

    aggregate_results['epoch'] = epoch
    if aggregate_results["overall"] > best_score:
        # print("new best model saved")
        best_score = aggregate_results["overall"]
        best_score_epoch = epoch

        if not os.path.exists('output'):
            os.makedirs('output')
        torch.save(model.state_dict(), osp.join('output', "model.pth"))
        if wandb_run:
            wandb.save("model.pth")

    return best_score, best_score_epoch


def train(model, configs, train_dataset, test_dataset, wandb_run=False):
    errors = []
    train_loader = DataLoader(train_dataset, batch_size=configs['batch_size'], shuffle=True)
    if 'Swin' in configs['model']:
        model = model(configs=configs).cuda()
    else:
        model = model.cuda()

    file_dict = {'discrete_classification': "pretrain_weight_classification.pkl",
                 'denoise': "pretrain_weight_denoise.pkl"}
    if configs['pretrain'] is not None:
        weight_path = osp.join(osp.dirname(osp.abspath(__file__)), "pretrain", "weights", configs['model'],
                               file_dict[configs['pretrain']])

        if os.path.exists(weight_path):
            pre_weights = torch.load(weight_path, map_location=torch.device("cuda"))
            for name, param in model.named_parameters():
                if name in pre_weights:
                    param.data.copy_(pre_weights[name])

    optimizer = optim.AdamW(model.parameters(), lr=configs['lr'], betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=configs['weight_decay'])
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

        if epoch % 1 == 0:
            best_score, best_score_epoch = valid(model, test_dataset, best_score, best_score_epoch, epoch, wandb_run)

    return {"best_score": best_score, "best_score_epoch": best_score_epoch, "best_loss": best_loss}


if __name__ == '__main__':

    configs = {
        'pretrain': None,
        'img_size': 512,
        'model': 'Resnet18',
        'epochs': 50,
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

    imgs_list, label_list = create_datalists()

    left_bound, right_bound = 900, 1000

    one_patient_out = True
    if one_patient_out:
        patient_indices = [0, 1, 3, 8, 20, 24, 29, 33, 36, 41, 42, 58, 79, 88, 90, 93, 97, 109, 112, 128, 132, 133, 134,
                           137, 155, 168, 187, 191, 208, 210, 220, 227, 228, 231, 234, 241, 249, 252, 256, 257, 259,
                           264, 267,
                           277, 284, 286, 287, 294, 299, 308, 314, 318, 321, 326, 328, 341, 349, 363, 365, 370, 372,
                           380, 397, 404, 405, 407, 412,
                           429, 440, 446, 448, 455, 457, 464, 465, 469, 481, 489, 491, 498,
                           509, 517, 518, 520, 527, 530, 533, 538, 546, 547, 553, 564, 565, 568, 569, 589,
                           591, 608, 609, 610, 612, 621, 638, 643, 653, 668, 669, 671, 674, 677, 685, 691,
                           699, 701, 714, 728, 729, 740, 742, 743, 746, 754, 759, 761, 764, 766, 767, 790,
                           791, 796, 818, 825, 837, 855, 859, 866, 868, 877, 888, 892, 904, 907, 910, 914,
                           942, 943, 945, 965, 980, 996
                           ]
        print(len(patient_indices))
        non_patient_indices = list(set(list(range(1000))) - set(patient_indices))
        train_dataset = CT_Dataset([imgs_list[x] for x in non_patient_indices], [label_list[x] for x in non_patient_indices], split="train",
                                   config=configs)
        test_dataset = CT_Dataset([imgs_list[x] for x in patient_indices], [label_list[x] for x in patient_indices], split="test", config=configs)
    else:

        train_dataset = CT_Dataset(imgs_list[:left_bound] + imgs_list[right_bound:],
                                   label_list[:left_bound] + label_list[right_bound:], split="train", config=configs)
        test_dataset = CT_Dataset(imgs_list[left_bound:right_bound], label_list[left_bound:right_bound], split="test",
                                  config=configs)

    train(model, configs, train_dataset, test_dataset, wandb_run=False)
