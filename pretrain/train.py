import os
import random
import sys
import time
import json
import torch
import torch.optim as optim
import numpy as np
import albumentations as A
from glob import glob
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
import os.path as osp
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import torch.nn as nn
from pretrain_models.model_efficientnet_denoise import Efficient_Swin_Denoise
from pretrain_models.model_resnet_denoise import Resnet34_Swin_Denoise

from models.res34_swin import Resnet34_Swin
from models.res34_swinv2 import Resnet34_Swinv2
from models.efficient_swin import Efficientnet_Swin
from models.efficient_swinv2 import Efficientnet_Swinv2

from measure import compute_PSNR, compute_SSIM
from warmup_scheduler.scheduler import GradualWarmupScheduler

from pretrain_dataloaders.aapm_dataset import AAPMDataset
from pretrain_dataloaders.classic_dataset import CT_Dataset_v1

from util.create_dataset import create_datasets, create_aapm_dataset

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
    if parameters["folder"] == "denoise_task_2K":
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
            for j in range(len(names)):
                np.save(names[j], imgs[j])
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


# training_data, given_params, context are necessary to make UbiOps work
def train(training_data, parameters, context):
    denoise_models = {"Resnet34_Swin": Resnet34_Swin_Denoise, "Efficientnet_Swin": Efficient_Swin_Denoise}
    classify_models = {"Resnet34_Swin": Resnet34_Swin, "Resnet34_Swinv2": Resnet34_Swinv2,
              "Efficientnet_Swin": Efficientnet_Swin, "Efficientnet_Swinv2": Efficientnet_Swinv2}

    if parameters['datasets'] == 'Classic':

        train_dataset, test_dataset = create_datasets(parameters)
    else:

        pretrain_path = osp.dirname(__file__)
        train_dataset, test_dataset = create_aapm_dataset(pretrain_path)

    # print(train_dataset[0])

    train_loader = DataLoader(train_dataset, batch_size=parameters["batch_size"], shuffle=True, num_workers=6)
    if parameters["folder"] == "denoise_task_2K":
        model = denoise_models[parameters["model_name"]]().to("cuda")
    else:
        model = classify_models[parameters["model_name"]](configs=parameters, out_channel=4).to("cuda")

    epochs = parameters["epochs"]
    optimizer = optim.AdamW(model.parameters(), lr=parameters["lr"], betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=parameters["weight_decay"])
    warmup_epochs = parameters["warmup_epochs"]
    nepoch = parameters["nepoch"]
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, nepoch - warmup_epochs,
                                                            eta_min=parameters["min_lr"])
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                       after_scheduler=scheduler_cosine)


    # num_steps = len(train_loader) * given_params["epochs"]
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=given_params["min_lr"])
    # warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)

    for epoch in range(epochs + 1):  # , colour='yellow', leave=False, position=0):
        start_time = time.time()
        losses = 0
        model.train()

        t = tqdm(enumerate(train_loader), total=len(train_loader), desc="epoch " + f"{epoch:04d}", colour='cyan')
        for i, (image, target) in t:
            image = image.to("cuda")
            target = target.to("cuda")
            pred = model(image)

            if parameters["folder"] == "denoise_task_2K":
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

        # print("epoch:", epoch, "loss:", float(losses / len(train_dataset)), f"time: {formatted_time}")
        #
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
        "folder": "denoise_task_2K",  # weighted_dataset, denoise_task_2K
        "split_ratio": 0.8,
        "batch_size": 4,
        "warmup_epochs": 20,
        "epochs": 500,
        "nepoch": 500,
        "lr": 3e-4,
        "min_lr": 1e-6,
        "weight_decay": 1e-4,
        "model_name": "Resnet34_Swin",  # Resnet34_Swin, Resnet34_Swinv2, Efficientnet_Swin, Efficientnet_Swinv2
        "img_size": 512,
        "use_avg": True,
        "use_mix": True,
        "datasets": "AAPM"
    }

    # model_names = ["Resnet34_Swin"]
    model_names = ["Resnet34_Swin", "Resnet34_Swinv2", "Efficientnet_Swin", "Efficientnet_Swinv2"]
    for m in model_names:
        parameters["model_name"] = m
        train(None, parameters, None)

