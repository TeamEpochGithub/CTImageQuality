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
from pretrain.train import create_datasets, set_seed
from warmup_scheduler.scheduler import GradualWarmupScheduler

set_seed(0)


# Define your LightningModule
class LightningModule(pl.LightningModule):
    def __init__(self, parameters, model):
        super(LightningModule, self).__init__()
        self.model = model
        self.parameters = parameters
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
        img = inputs.unsqueeze(0).float()
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
        # outputs = self(inputs)

        # self.log('val_loss', loss)
        # Calculate validation metrics
        # self.log('val_metric', metric_value)
        return {'loss': loss, 'psnrs': psnrs, "ssims": ssims}

    def validation_epoch_end(self, outputs):
        # Calculate the average loss across all validation batches
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_psnr = torch.stack([x['psnrs'] for x in outputs]).mean()
        avg_ssim = torch.stack([x['ssims'] for x in outputs]).mean()
        # Log the average loss
        self.log('val_loss', avg_loss, on_epoch=True, prog_bar=True)
        self.log('val_psnr', avg_psnr, on_epoch=True, prog_bar=True)
        self.log('val_ssim', avg_ssim, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.parameters["lr"], betas=(0.9, 0.999), eps=1e-8,
                                weight_decay=self.parameters["weight_decay"])
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                self.parameters["nepoch"] - self.parameters[
                                                                    "warmup_epochs"],
                                                                eta_min=self.parameters["min_lr"])
        scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=self.parameters["warmup_epochs"],
                                           after_scheduler=scheduler_cosine)
        return [optimizer, scheduler]


def train(training_data, parameters, context):
    train_dataset, test_dataset = create_datasets(parameters)
    train_loader = DataLoader(train_dataset, batch_size=parameters["batch_size"], shuffle=True)
    valid_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="val_psnr",
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

    module = LightningModule(parameters, model)
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    return str(trainer.callback_metrics['val/loss'].item())


if __name__ == '__main__':
    parameters = {
        "split_ratio": 0.8,
        "batch_size": 8,
        "warmup_epochs": 20,
        "epochs": 100000,
        "nepoch": 500,
        "lr": 3e-4,
        "min_lr": 1e-6,
        "weight_decay": 1e-4,
        "model_name": "efficientnet"
    }
    train(None, parameters, None)
