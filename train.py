import os
import os.path as osp
import random
import json
import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from datasets import CT_Dataset
from models.res34_swin import Unet34_Swin
import pytorch_warmup as warmup
from scipy.stats import pearsonr, spearmanr, kendalltau
import tifffile
from PIL import Image
import LDCTIQAG2023_train as train_data
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy

import logging
logging.getLogger("lightning.pytorch").setLevel(logging.DEBUG)

torch.set_float32_matmul_precision('medium')

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
    "batch_size": 16,
    "epochs": 251,
    "lr": 3e-4,
    "min_lr": 1e-6,
    "weight_decay": 1e-4
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


def valid(model, test_dataset):
    model.eval()
    total_pred = []
    total_gt = []
    aggregate_results = dict()
    best_score = 0
    with torch.no_grad():
        for _, (img, label) in enumerate(test_dataset):
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
    print("validation metrics:", aggregate_results)
    if aggregate_results["overall"] > best_score:
        best_score = aggregate_results["overall"]
        torch.save(model.state_dict(), "swin_model.pth")


def train(model):

    logger = TensorBoardLogger(save_dir="logs/logs")

    split_num = 900
    train_dataset = CT_Dataset(imgs_list[:split_num], label_list[:split_num], split="train")
    test_dataset = CT_Dataset(imgs_list[split_num:], label_list[split_num:], split="test")
    train_loader = DataLoader(train_dataset, batch_size=configs["batch_size"], shuffle=True, num_workers=8)
    valid_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

    strategy = DDPStrategy(process_group_backend="gloo", find_unused_parameters=True)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    trainer = Trainer(
        max_epochs=500,
        logger=True,
        log_every_n_steps=50,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,
        accelerator='gpu',
        devices=1,
        strategy=strategy,
        precision='16-mixed',
        enable_progress_bar=True,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    return model


class BaseModule(pl.LightningModule):
    def __init__(self, model, epochs, warmup_epochs, learning_rate, weight_decay, loss_function):
        super().__init__()
        self.model = model
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_function = loss_function

        self.optimizer = optim.AdamW(model.parameters(), lr=configs["lr"], betas=(0.9, 0.999), eps=1e-8,
                                     weight_decay=configs["weight_decay"])

        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                       self.epochs - self.warmup_epochs,
                                                                       eta_min=1e-6)
        self.warmup_scheduler = warmup.UntunedLinearWarmup(self.optimizer)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x).squeeze()
        loss = self.loss_function(y_hat, y)
        self.log("train_loss", loss, sync_dist=True, prog_bar=True, on_epoch=True)
        self.optimizer.step()
        with self.warmup_scheduler.dampening():
            self.lr_scheduler.step()

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x).squeeze()
        loss = self.loss_function(y_hat, y)
        self.log("val_loss", loss, sync_dist=True, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return [self.optimizer], [self.lr_scheduler]

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = Unet34_Swin()

    module = BaseModule(model, 250, 20, configs["lr"], configs["weight_decay"], F.mse_loss)

    train(module)
