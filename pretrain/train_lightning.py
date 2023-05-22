import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from measure import compute_PSNR, compute_SSIM
from model_efficientnet import Efficient_Swin
from model_resnet import Resnet34_Swin
from train import create_datasets, set_seed
from warmup_scheduler.scheduler import GradualWarmupScheduler

set_seed(0)


# Define your LightningModule
class LightningModule(pl.LightningModule):
    def __init__(self, model):
        super(LightningModule, self).__init__()
        self.model = model
        self.loss_fn = F.mse_loss

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_fn(inputs - outputs, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        img = inputs.float()
        noise = self.model(img)
        pred = img - noise
        loss = self.loss_fn(pred, labels)
        pred = pred.cpu()
        pred_new = pred.numpy().squeeze(0)
        pred_new = pred_new.reshape(512, 512)

        label_new = labels.cpu().numpy()
        label_new = label_new.reshape(512, 512)

        psnrs = compute_PSNR(label_new, pred_new, data_range=1)
        ssims = compute_SSIM(labels, pred, data_range=1)

        self.log('val_loss', loss)
        self.log("val_psnr", psnrs)
        self.log("val_ssim", ssims)
        # Calculate validation metrics
        # self.log('val_metric', metric_value)
        return loss, psnrs, ssims


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-8,
                                weight_decay=1e-4)
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                480,
                                                                eta_min=1e-6)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=20,
                                           after_scheduler=scheduler_cosine)
        return [optimizer], [scheduler]


def train(training_data, parameters, context):
    train_dataset, test_dataset = create_datasets(parameters)
    train_loader = DataLoader(train_dataset, batch_size=parameters["batch_size"], shuffle=True)
    valid_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="denoise",
        mode="min",
    )

    # ddp = DDPStrategy(process_group_backend="gloo")
    trainer = Trainer(
        max_epochs=parameters["epochs"],
        log_every_n_steps=10,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,
        accelerator='gpu',
        devices=1,
        # num_nodes=4,
        # strategy=ddp
    )

    if parameters["model_name"] == "resnet":
        model = Resnet34_Swin()
    else:
        model = Efficient_Swin()

    module = LightningModule(model)
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    return str(trainer.callback_metrics['val_loss'].item())


if __name__ == '__main__':
    parameters = {
        "split_ratio": 0.8,
        "batch_size": 2,
        "warmup_epochs": 20,
        "epochs": 100000,
        "nepoch": 500,
        "lr": 3e-4,
        "min_lr": 1e-6,
        "weight_decay": 1e-4,
        "model_name": "efficientnet"
    }
    train(None, parameters, None)
