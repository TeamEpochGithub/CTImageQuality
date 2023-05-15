from dataloader import CTIDataset
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torchvision.models import resnet34, resnet152, efficientnet_b0, efficientnet_b7, efficientnet_b2
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.model_selection import train_test_split
import json
import pytorch_warmup as warmup
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


def valid(model, device, criterion, val_dataloader):
    # Validation
    model.eval()
    val_loss = 0.0
    total_pred = []
    total_gt = []

    with torch.no_grad():
        for images, labels in val_dataloader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            total_pred.extend(outputs.squeeze().cpu().numpy())
            total_gt.extend(labels.squeeze().cpu().numpy())

    val_loss = val_loss / len(val_dataloader)

    total_pred = np.array(total_pred)
    total_gt = np.array(total_gt)

    aggregate_results = dict()
    aggregate_results["plcc"] = abs(pearsonr(total_pred, total_gt)[0])
    aggregate_results["srocc"] = abs(spearmanr(total_pred, total_gt)[0])
    aggregate_results["krocc"] = abs(kendalltau(total_pred, total_gt)[0])
    aggregate_results["overall"] = abs(pearsonr(total_pred, total_gt)[0]) + abs(
        spearmanr(total_pred, total_gt)[0]) + abs(kendalltau(total_pred, total_gt)[0])

    return val_loss, aggregate_results


def adapt_resnet_to_grayscale(model):
    num_channels = 1  # Grayscale images have 1 channel
    model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model


def adapt_efficientnet_to_grayscale(model):
    num_channels = 1  # Grayscale images have 1 channel
    out_channels = model.features[0][0].out_channels
    kernel_size = model.features[0][0].kernel_size
    stride = model.features[0][0].stride
    padding = model.features[0][0].padding
    model.features[0][0] = nn.Conv2d(num_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                     padding=padding, bias=False)
    return model


def train(img_dir, label_dir, configs):
    with open(label_dir, 'r') as f:
        labels = json.load(f)
    file_names = list(labels.keys())

    train_file_names, val_file_names = train_test_split(file_names, test_size=0.1, random_state=42)

    train_dataset = CTIDataset(img_dir=img_dir, labels=labels, file_names=train_file_names)
    val_dataset = CTIDataset(img_dir=img_dir, labels=labels, file_names=val_file_names)

    train_dataloader = DataLoader(train_dataset, batch_size=configs["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=configs["batch_size"], shuffle=True)

    # Instantiate the ResNet model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = efficientnet_b0(pretrained=True)
    model = adapt_efficientnet_to_grayscale(model)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    print(model)

    # Use pretrained weights for transfer learning
    model = model.to(device)
    # Set up the loss function and optimizer
    criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.AdamW(model.parameters(), lr=configs["lr"], betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=configs["weight_decay"])
    num_steps = len(train_dataloader) * configs["epochs"]
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=configs["min_lr"])
    warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
    # Training loop
    num_epochs = configs["epochs"]
    best_score = 0
    best_epoch = 0

    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        # pbar = tqdm(enumerate(dataloader), total=len(dataloader), ncols=100)
        for batch_idx, (images, labels) in enumerate(train_dataloader):
            images, labels = images.to(device), labels.to(device).unsqueeze(1)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            with warmup_scheduler.dampening():
                lr_scheduler.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_dataloader)
        val_loss, aggregate_results = valid(model, device, criterion, val_dataloader)
        eval_score = aggregate_results['overall']
        if eval_score > best_score:
            best_score = eval_score
            best_epoch = epoch + 1
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss}, Validation Loss: {val_loss}, Validation Metrics: {aggregate_results}, lr: {lr_scheduler.get_last_lr()}")

        wandb.log({"Train Loss": train_loss, "Validation Loss": val_loss, "Validation Metrics": aggregate_results['overall'], " Best score": best_score}, step=epoch+1)

    print("#### FINISHED TRAINING ####")
    print(f"BEST SCORE: {best_score} at epoch {best_epoch}")

    wandb.finish()


if __name__ == "__main__":
    img_dir = r'C:\EpochProjects\CTImageQuality\data\LDCTIQAG2023_train\image'
    label_json = r'C:\EpochProjects\CTImageQuality\data\LDCTIQAG2023_train\train.json'

    configs = {
        "batch_size": 8,
        "epochs": 251,
        "lr": 3e-4,
        "min_lr": 1e-6,
        "weight_decay": 1e-4

    }

    wandb.init(
        # set the wandb project where this run will be logged
        project="eerste test",

        # track hyperparameters and run metadata
        config={
            "architecture": "EfficientNet-B0",
            "batch_size": configs['batch_size'],
            "epochs": configs['epochs'],
            "learning_rate": configs['lr'],
            "weight_decay": configs["weight_decay"]
        }
    )

    train(img_dir, label_json, configs)
