import os
import os.path as osp
import random
import json
import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import CT_Dataset
from models.res34_swin import Unet34_Swin
from models.res34_swinv2 import Unet34_Swinv2
from models.efficient_swinv2 import Efficientnet_Swinv2
from models.efficient_swin import Efficientnet_Swin
import pytorch_warmup as warmup
from scipy.stats import pearsonr, spearmanr, kendalltau
import tifffile
from PIL import Image
import LDCTIQAG2023_train as train_data
import wandb

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

configs = {
    "batch_size": 32,
    "epochs": 11,
    "lr": 5e-4,
    "min_lr": 1e-6,
    "weight_decay": 1e-4,
    "split_num": 900,
}

data_dir = osp.join(osp.dirname(train_data.__file__), 'image')
label_dir = osp.join(osp.dirname(train_data.__file__), 'train.json')
with open(label_dir, 'r') as f:
    label_dict = json.load(f)

imgs_list = []
label_list = []
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith('.tif'):
            label_list.append(label_dict[file])
            with tifffile.TiffFile(os.path.join(root, file)) as tif:
                image = tif.pages[0].asarray()
                img = Image.fromarray(image)
                imgs_list.append(img)


def valid(model, test_dataset, best_score):
    model.eval()
    total_pred = []
    total_gt = []
    aggregate_results = dict()
    with torch.no_grad():
        for _, (img, label) in tqdm(enumerate(test_dataset), desc="Validation", total=1000-configs["split_num"], colour='blue'):
            img = img.unsqueeze(0).float()
            pred = model(img.cuda())
            pred_new = pred.cpu().numpy().squeeze(0)
            label_new = label.cpu().numpy()
            total_pred.append(pred_new[0])
            total_gt.append(label_new)
        total_pred = np.array(total_pred)
        total_gt = np.array(total_gt)
        aggregate_results["plcc"] = abs(pearsonr(total_pred, total_gt)[0])
        aggregate_results["srocc"] = abs(spearmanr(total_pred, total_gt)[0])
        aggregate_results["krocc"] = abs(kendalltau(total_pred, total_gt)[0])
        aggregate_results["overall"] = abs(pearsonr(total_pred, total_gt)[0]) + abs(
            spearmanr(total_pred, total_gt)[0]) + abs(kendalltau(total_pred, total_gt)[0])
    print("validation metrics:", {key: round(value, 3) for key, value in aggregate_results.items()})
    wandb.log(aggregate_results)
    if aggregate_results["overall"] > best_score:
        print("new best model saved")
        best_score = aggregate_results["overall"]

        if not os.path.exists('output'):
            os.makedirs('output')
        torch.save(model.state_dict(), osp.join('output', "model.pth"))
        wandb.save("model.pth")

    return best_score


def train(model):
    train_dataset = CT_Dataset(imgs_list[:configs["split_num"]], label_list[:configs["split_num"]], split="train")
    test_dataset = CT_Dataset(imgs_list[configs["split_num"]:], label_list[configs["split_num"]:], split="test")
    train_loader = DataLoader(train_dataset, batch_size=configs["batch_size"], shuffle=True)
    model = model().cuda()  # model = Efficient_Swinv2_Next().cuda()
    # model = nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=configs["lr"], betas=(0.9, 0.999), eps=1e-8, weight_decay=configs["weight_decay"])
    num_steps = len(train_loader) * configs["epochs"]
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=configs["min_lr"])
    warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)

    best_score = 0
    for epoch in range(configs["epochs"]):
        losses = 0
        model.train()
        for _, (image, target) in tqdm(enumerate(train_loader), desc="Training", total=len(train_loader), colour='green'):
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

        loss = float(losses / len(train_dataset))
        print("epoch:", epoch, "loss:", loss, "lr:", lr_scheduler.get_last_lr())
        wandb.log({"loss": loss, "epoch": epoch})

        if epoch % 2 == 0:
            best_score = valid(model, test_dataset, best_score)


if __name__ == "__main__":
    wandb.login()

    run = wandb.init(
        project="CTImageQuality-regression",
        notes="My first experiment",
        tags=["baselines"],
        config=configs,
    )

    models = [Unet34_Swin, Unet34_Swinv2, Efficientnet_Swin, Efficientnet_Swinv2]

    for model in models:
        train(model)
