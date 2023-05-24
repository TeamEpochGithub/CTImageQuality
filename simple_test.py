import numpy as np
import torch
import torchvision
from PIL import Image
import os
import os.path as osp
import random
import torch
import torch.optim as optim
import json
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import LDCTIQAG2023_train as train_data
from models.res34_swin import Resnet34_Swin
from models.res34_swinv2 import Resnet34_Swinv2
from models.efficient_swinv2 import Efficientnet_Swinv2
from models.efficient_swin import Efficientnet_Swin
import pytorch_warmup as warmup
from scipy.stats import pearsonr, spearmanr, kendalltau
import tifffile

class CT_Dataset(torch.utils.data.Dataset):
    def __init__(self, imgs_list, label_list, image_size, split):
        self.imgs_list = imgs_list
        self.label_list = label_list
        self.split = split
        self.image_size = image_size

        if self.split == 'train':

            operations = [torchvision.transforms.ToPILImage()]

            operations.append(torchvision.transforms.RandomHorizontalFlip())

            operations.append(torchvision.transforms.RandomVerticalFlip())

            operations.append(torchvision.transforms.RandomRotation(15))

            operations += [torchvision.transforms.ToTensor()]

            self.transform = torchvision.transforms.Compose(operations)

        else:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        x = self.imgs_list[idx]
        x = x.resize((self.image_size, self.image_size), Image.LANCZOS)
        x = np.array(x)
        x = self.transform(x)
        y = self.label_list[idx]
        return x, torch.tensor(y)


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

def create_datalists():
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

    return imgs_list, label_list

def valid(model, test_dataset, best_score, best_score_epoch, epoch):
    model.eval()
    total_pred = []
    total_gt = []
    aggregate_results = dict()
    with torch.no_grad():
        for _, (img, label) in tqdm(enumerate(test_dataset), desc="Validation", total=len(test_dataset),
                                    colour='blue'):
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

    aggregate_results['epoch'] = epoch
    if aggregate_results["overall"] > best_score:
        print("new best model saved")
        best_score = aggregate_results["overall"]
        best_score_epoch = epoch

        if not os.path.exists('output'):
            os.makedirs('output')
        torch.save(model.state_dict(), osp.join('output', "model.pth"))

    return best_score, best_score_epoch


def train(model, configs):
    imgs_list, label_list = create_datalists()
    train_dataset = CT_Dataset(imgs_list[:900], label_list[:900], split="train", image_size = configs["image_size"])
    test_dataset = CT_Dataset(imgs_list[900:], label_list[900:], split="test", image_size = configs["image_size"])
    train_loader = DataLoader(train_dataset, batch_size=configs['batch_size'], shuffle=True)
    model = model(configs=configs).cuda()

    optimizer = optim.AdamW(model.parameters(), lr=configs["lr"], betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=configs["weight_decay"])
    num_steps = len(train_loader) * configs["epochs"]
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=configs["min_lr"])
    warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)

    best_score = 0
    best_loss = 10
    best_score_epoch = 0
    for epoch in range(configs["epochs"]):
        losses = 0
        model.train()
        for _, (image, target) in tqdm(enumerate(train_loader), desc="Training", total=len(train_loader),
                                       colour='green'):
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
        if loss < best_loss:
            best_loss = loss
        print("epoch:", epoch, "loss:", loss, "lr:", lr_scheduler.get_last_lr())

        if epoch % 1 == 0:
            best_score, best_score_epoch = valid(model, test_dataset, best_score, best_score_epoch, epoch)

    return {"best_score": best_score, "best_score_epoch": best_score_epoch, "best_loss": best_loss}

if __name__ == '__main__':
    image_size = 256
    resnet_swin_config = {
        "batch_size": 2,
        "epochs": 200,
        "lr": 3e-4,
        "min_lr": 1e-6,
        "weight_decay": 1e-4,
        'img_size': image_size,
        'use_avg': True,
        'use_mix': True
    }
    train(Resnet34_Swin, resnet_swin_config)
