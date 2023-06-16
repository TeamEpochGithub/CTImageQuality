import os
import random
import time
import json
import torch
import torch.optim as optim
import numpy as np
import albumentations as A
from glob import glob

from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
import os.path as osp
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import torch.nn as nn
from pretrain_models.model_efficientnet_denoise import Efficient_Swin_Denoise
from pretrain_models.model_resnet_denoise import Resnet34_Swin_Denoise
from pretrain_models.resnet34_unet import UNet34_Denoise
from pretrain_models.efficientnet_unet import EfficientNet_Denoise

import sys
# sys.path.append('..')
# from models.resnet import load_resnet_model
# from models.efficientnet import load_efficientnet_model
# from models.res34_swin import Resnet34_Swin
# from models.res34_swinv2 import Resnet34_Swinv2
# from models.efficient_swin import Efficientnet_Swin
# from models.efficient_swinv2 import Efficientnet_Swinv2

from measure import compute_PSNR, compute_SSIM
from warmup_scheduler.scheduler import GradualWarmupScheduler

torch.cuda.set_device(0)


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


class CT_Dataset(Dataset):
    def __init__(self, lists, mode, norm, transform=None):
        self.lists = lists
        self.norm = norm
        self.mode = mode
        self.transform = transform
        self.target_ = []
        for i in range(len(self.lists)):
            self.target_.append(self.lists[i][1])

    def __len__(self):
        return len(self.lists)

    def __getitem__(self, idx):
        if self.mode == "denoise_task_2K" or self.mode == "AAPM":
            input_img, target_img = self.lists[idx]
            input_img, target_img = np.float32(np.load(input_img)), np.float32(np.load(target_img))
            if self.norm:
                input_img = (input_img - np.min(input_img)) / (np.max(input_img) - np.min(input_img))
                target_img = (target_img - np.min(target_img)) / (np.max(target_img) - np.min(target_img))
            augmentations = self.transform(image=input_img, mask=target_img)
            image = augmentations["image"]
            label = augmentations["mask"]
        else:
            input_img, label = self.lists[idx]
            input_img = np.float32(np.load(input_img))
            augmentations = self.transform(image=input_img)
            image = augmentations["image"]
            label = torch.tensor(label)
        return image, label


best_psnr = 0
best_ssim = 0
best_acc = 0


def test(parameters, model, test_dataset):
    pretrain_path = osp.dirname(__file__)

    global best_psnr
    global best_ssim
    global best_acc
    psnrs = []
    ssims = []
    imgs = []
    names = []

    save_path = osp.join(pretrain_path, 'weights', parameters["model_name"])

    if not osp.exists(save_path):
        os.mkdir(save_path)

    img_path = osp.join(pretrain_path, 'output_imgs')
    if not os.path.exists(img_path):
        os.mkdir(img_path)

    model.eval()
    if parameters["folder"] == "denoise_task_2K" or parameters["folder"] == "AAPM":
        with torch.no_grad():
            for i, (img, label) in tqdm(enumerate(test_dataset), total=len(test_dataset), desc="testing: ",
                                        colour='blue'):
                img = img.unsqueeze(0).float().to("cuda")
                noise = model(img)
                pred = img - noise
                pred = pred.cpu()
                pred_new = pred.numpy().squeeze(0)
                pred_new = pred_new.reshape(512, 512)

                label_new = label.cpu().numpy()
                label_new = label_new.reshape(512, 512)

                img_name = test_dataset.target_[i]
                image_name = img_name.split("\\")[-1]
                out_path = os.path.join(img_path, image_name)
                names.append(out_path)
                imgs.append(pred_new)

                psnrs.append(compute_PSNR(label_new, pred_new, data_range=1))
                ssims.append(compute_SSIM(label, pred, data_range=1))

        pt = np.mean(np.array(psnrs))
        st = np.mean(np.array(ssims))
        print("PSNR:", round(pt, 3))
        print("SSIM:", round(st, 3))

        if pt > best_psnr and st > best_ssim:
            best_psnr = pt
            best_ssim = st
            path_file = os.path.join(save_path, "pretrain_weight_denoise.pkl")
            torch.save(model.state_dict(), path_file)
            # for j in range(len(names)):
            #     np.save(names[j], imgs[j])
        print("best PSNR:", round(best_psnr, 3))
        print("best SSIM:", round(best_ssim, 3))
    else:
        preds = []
        labels = []
        with torch.no_grad():
            for i, (img, label) in tqdm(enumerate(test_dataset), total=len(test_dataset), desc="testing: ",
                                        colour='blue'):
                img = img.unsqueeze(0).float().to("cuda")
                pred = model(img)
                pred = pred.cpu().numpy()
                pred = np.argmax(pred[0])
                preds.append(pred)
                label = label.cpu().numpy()
                labels.append(label)
        print("preds:", preds[:10])
        print("labels:", labels[:10])
        accuracy = accuracy_score(labels, preds)
        print("testing accuracy:", accuracy)
        if accuracy > best_acc:
            best_acc = accuracy
            path_file = os.path.join(save_path, "pretrain_weight_classification.pkl")
            torch.save(model.state_dict(), path_file)


def create_datasets(parameters):
    pretrain_path = osp.dirname(__file__)
    folder = parameters["folder"]
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        ToTensorV2()
    ])

    test_transform = A.Compose([
        ToTensorV2()
    ])

    data_path = osp.join(pretrain_path, 'pretrain_data', folder)
    lists = []
    if folder == "denoise_task_2K":
        input_path = sorted(glob(os.path.join(data_path, '*input*.npy')))
        target_path = sorted(glob(os.path.join(data_path, '*target*.npy')))
        for i in range(len(input_path)):
            lists.append((input_path[i], target_path[i]))
    elif folder == "AAPM":
        train_FD_path = sorted(glob(os.path.join(data_path, "train_set", 'FD_NPY', '*FD*.npy')))
        train_QD_path = sorted(glob(os.path.join(data_path, "train_set", 'QD_NPY', '*QD*.npy')))
        test_FD_path = sorted(glob(os.path.join(data_path, "test_set", 'FD_NPY', '*FD*.npy')))
        test_QD_path = sorted(glob(os.path.join(data_path, "test_set", 'QD_NPY', '*QD*.npy')))
    else:
        labels = osp.join(data_path, 'label.json')
        with open(labels, 'r') as f:
            label_dict = json.load(f)
        for key in label_dict:
            lists.append((osp.join(data_path, "image", key), label_dict[key]))

    train_lists = []
    test_lists = []
    if folder == "AAPM":
        for i in range(len(train_FD_path)):
            train_lists.append((train_QD_path[i], train_FD_path[i]))
        for i in range(len(test_FD_path)):
            test_lists.append((test_QD_path[i], test_FD_path[i]))
        train_dataset = CT_Dataset(train_lists, transform=train_transform, norm=False, mode=folder)
        test_dataset = CT_Dataset(test_lists, transform=test_transform, norm=False, mode=folder)
    else:
        random.shuffle(lists)
        train_lists = lists[:int(len(lists) * parameters["split_ratio"])]
        test_lists = lists[int(len(lists) * parameters["split_ratio"]):]
        train_dataset = CT_Dataset(train_lists, transform=train_transform, norm=True, mode=folder)
        test_dataset = CT_Dataset(test_lists, transform=test_transform, norm=True, mode=folder)
    return train_dataset, test_dataset


# training_data, given_params, context are necessary to make UbiOps work
def train(training_data, parameters, context):
    denoise_models = {"Resnet34_Swin": Resnet34_Swin_Denoise(), "Efficientnet_Swin": Efficient_Swin_Denoise(),
                      "ResNet34": UNet34_Denoise(), "Efficientnet_B0": EfficientNet_Denoise(mode="b0"),
                      "Efficientnet_B1": EfficientNet_Denoise(mode="b1"),
                      "Efficientnet_B2": EfficientNet_Denoise(mode="b2"),
                      "Efficientnet_B3": EfficientNet_Denoise(mode="b3"),
                      "Efficientnet_B4": EfficientNet_Denoise(mode="b4"),
                      "Efficientnet_B5": EfficientNet_Denoise(mode="b5"),
                      "Efficientnet_B6": EfficientNet_Denoise(mode="b6"),
                      "Efficientnet_B7": EfficientNet_Denoise(mode="b7"), }
    configs = {
        "pretrain": None
    }
    # classify_models = {'Resnet18': load_resnet_model('18', configs['pretrain'], out_channel=4),
    #           'Resnet50': load_resnet_model('50', configs['pretrain'], out_channel=4),
    #           'Resnet152': load_resnet_model('152', configs['pretrain'], out_channel=4),
    #           'Efficientnet_B0': load_efficientnet_model('b0', configs['pretrain'], out_channel=4),
    #           'Efficientnet_B4': load_efficientnet_model('b4', configs['pretrain'], out_channel=4),
    #           'Efficientnet_B7': load_efficientnet_model('b7', configs['pretrain'], out_channel=4),
    #           'Efficientnet_Swin': Efficientnet_Swin(configs=parameters, out_channel=4),
    #           'Efficientnet_Swinv2': Efficientnet_Swinv2(configs=parameters, out_channel=4),
    #           'Resnet34_Swin': Resnet34_Swin(configs=parameters, out_channel=4),
    #           'Resnet34_Swinv2': Resnet34_Swinv2(configs=parameters, out_channel=4)}

    train_dataset, test_dataset = create_datasets(parameters)

    train_loader = DataLoader(train_dataset, batch_size=parameters["batch_size"], shuffle=True, num_workers=4)
    if parameters["folder"] == "denoise_task_2K" or parameters["folder"] == "AAPM":
        model = denoise_models[parameters["model_name"]].to("cuda")
    # else:
    #     model = classify_models[parameters["model_name"]].to("cuda")

    epochs = parameters["epochs"]
    optimizer = optim.AdamW(model.parameters(), lr=parameters["lr"], betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=parameters["weight_decay"])
    warmup_epochs = parameters["warmup_epochs"]
    nepoch = parameters["nepoch"]
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, nepoch - warmup_epochs,
                                                            eta_min=parameters["min_lr"])
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                       after_scheduler=scheduler_cosine)

    for epoch in range(epochs + 1):  # , colour='yellow', leave=False, position=0):
        start_time = time.time()
        losses = 0
        model.train()

        t = tqdm(enumerate(train_loader), total=len(train_loader), desc="epoch " + f"{epoch:04d}", colour='cyan')
        for i, (image, target) in t:
            image = image.to("cuda")
            target = target.to("cuda")
            pred = model(image)

            if parameters["folder"] == "denoise_task_2K" or parameters["folder"] == "AAPM":
                loss_function = nn.MSELoss()
                target = target.unsqueeze(1)
                loss = loss_function(pred, image - target)
            else:
                loss_function = nn.CrossEntropyLoss()
                loss = loss_function(pred, target)

            losses += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if i == len(train_loader) - 1:
                t.set_postfix(
                    {"loss": round(float(losses / len(train_dataset)), 5), "lr": round(scheduler.get_lr()[0], 8)})

        end_time = time.time()
        execution_time = end_time - start_time
        minutes = int(execution_time // 60)
        seconds = int(execution_time % 60)
        formatted_time = f"{minutes:02d}:{seconds:02d}"

        print("epoch:", epoch, "loss:", float(losses / len(train_dataset)), f"time: {formatted_time}")

        if epoch % 15 == 0:
            test(parameters, model, test_dataset)

    return {
        "artifact": "None",
        "metadata": {},
        "metrics": {"no_metric": -1},
        "additional_output_files": []
    }


if __name__ == '__main__':
    parameters = {
        "folder": "AAPM",  # weighted_dataset, denoise_task_2K, AAPM
        "split_ratio": 0.8,
        "batch_size": 16,
        "warmup_epochs": 20,
        "epochs": 250,
        "nepoch": 250,
        "lr": 3e-4,
        "min_lr": 1e-6,
        "weight_decay": 0.03,
        "model_name": "Efficientnet_B6",  # ResNet34, Resnet34_Swin, Resnet34_Swinv2, Efficientnet_Swin, Efficientnet_Swinv2
        "img_size": 512,
        "use_avg": True,
        "use_mix": True,
    }

    # denoise for keys of denoise_models, while classification for keys of classify_models (recomand to use AAPM for denoise task)
    model_names = ["Efficientnet_B2"]  # ["Efficientnet_B1", "Efficientnet_B2", "Efficientnet_B3", "Efficientnet_B4", "Efficientnet_B5", "Efficientnet_B6",

    # Resnet34_Swin, ResNet34, Efficientnet_Swin
    for m in model_names:
        parameters["model_name"] = m
        train(None, parameters, None)
