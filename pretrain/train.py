import os
import random
import torch
import torch.optim as optim
import numpy as np
import albumentations as A
from glob import glob
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
import os.path as osp
from model import Resnet34_Swin
import pytorch_warmup as warmup
from measure import compute_PSNR, compute_SSIM

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
pretrain_path = osp.abspath(osp.dirname(osp.dirname(__file__)))

class CT_Dataset(Dataset):
    def __init__(self, mode, saved_path, test_patient="test", norm=True, transform=None):
        assert mode in ['train', 'test'], "mode is 'train' or 'test'"

        input_path = sorted(glob(os.path.join(saved_path, '*input*.npy')))
        target_path = sorted(glob(os.path.join(saved_path, '*target*.npy')))
        self.transform = transform
        self.norm = norm

        if mode == "train":
            input_ = [f for f in input_path if test_patient not in f]
            target_ = [f for f in target_path if test_patient not in f]
            self.input_ = input_
            self.target_ = target_
        elif mode == "test":
            input_ = [f for f in input_path if test_patient in f]
            target_ = [f for f in target_path if test_patient in f]
            self.input_ = input_
            self.target_ = target_

    def __len__(self):
        return len(self.target_)

    def __getitem__(self, idx):
        input_img, target_img = self.input_[idx], self.target_[idx]
        input_img, target_img = np.float32(np.load(input_img)), np.float32(np.load(target_img))
        if self.norm:
            input_img = (input_img-np.min(input_img))/(np.max(input_img)-np.min(input_img))
            target_img = (target_img-np.min(target_img))/(np.max(target_img)-np.min(target_img))
        augmentations = self.transform(image=input_img, mask=target_img)
        image = augmentations["image"]
        label = augmentations["mask"]
        return image, label

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
    ToTensorV2()
])

test_transform = A.Compose([
    ToTensorV2()
])

def statistics(path):
    target_path = sorted(glob(os.path.join(path, '*target*.npy')))
    mx = float("-inf")
    mn = float("inf")
    for f in target_path:
        img = np.load(f)
        mx = max(mx, np.max(img))
        mn = min(mn, np.min(img))
    return mx, mn


best_psnr = 0
best_ssim = 0
def test(model, test_dataset):
    global best_psnr
    global best_ssim
    psnrs = []
    ssims = []
    imgs = []
    names = []

    save_path = osp.join(pretrain_path, 'weights')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    img_path = osp.join(pretrain_path, 'output_imgs')
    if not os.path.exists(img_path):
        os.mkdir(img_path)

    model.eval()
    with torch.no_grad():
        for i, (img, label) in enumerate(test_dataset):
            img = img.unsqueeze(0).float().cuda()
            noise = model(img)
            pred = img-noise
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
    print("PSNR:", np.mean(np.array(psnrs)))
    print("SSIM:", np.mean(np.array(ssims)))

    pt = np.mean(np.array(psnrs))
    st = np.mean(np.array(ssims))

    if pt > best_psnr and st > best_ssim:
        best_psnr = pt
        best_ssim = st
        path_file = os.path.join(save_path, "pretrain_weight.pkl")
        torch.save(model.state_dict(), path_file)
        for j in range(len(names)):
            np.save(names[j], imgs[j])
    print("best PSNR:", best_psnr)
    print("best SSIM:", best_ssim)


def train(configs):
    data_path = osp.join(pretrain_path, 'npy_imgs')
    train_dataset = CT_Dataset("train", saved_path=data_path, transform=train_transform, norm=True)
    test_dataset = CT_Dataset("test", saved_path=data_path, transform=test_transform, norm=True)
    train_loader = DataLoader(train_dataset, batch_size=configs["batch_size"], shuffle=True)
    model = Resnet34_Swin().cuda()

    nepoch = configs["epochs"]
    optimizer = optim.AdamW(model.parameters(), lr=configs["lr"], betas=(0.9, 0.999), eps=1e-8, weight_decay=configs["weight_decay"])
    num_steps = len(train_loader) * configs["epochs"]
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=configs["min_lr"])
    warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)

    for epoch in range(nepoch + 1):
        losses = 0
        model.train()
        for i, (image, target) in enumerate(train_loader):
            image = image.cuda()
            target = target.unsqueeze(1).cuda()
            pred = model(image)
            loss = F.mse_loss(pred, image-target)
            losses += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with warmup_scheduler.dampening():
                lr_scheduler.step()
        print("epoch:", epoch, "loss:", float(losses / len(train_dataset)))

        if epoch % 15 == 0:
            test(model, test_dataset)

if __name__ == '__main__':
    configs = {
        "batch_size": 4,
        "epochs": 500,
        "lr": 3e-4,
        "min_lr": 1e-6,
        "weight_decay": 1e-4,
    }
    train(configs)
