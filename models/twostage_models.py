import torch
import torch.nn as nn
import os.path as osp
import sys
sys.path.append('..')
from pretrain.pretrain_models.efficientnet_unet import EfficientNet_Denoise
from pretrain.pretrain_models.edcnn import EDCNN
import torch.nn.functional as F
from models.edcnn_swin import EDCNN_Swin
import random
import models

root_dir = osp.join(osp.dirname(osp.dirname(models.__file__)), "pretrain", "weights")
weight_path = osp.join(root_dir, "Efficientnet_B1", "pretrain_weight_denoise.pkl")
edcnn_weight_path = osp.join(root_dir, "ED_CNN", "pretrain_weight_denoise.pkl")


class TwoStage_EfficientB0(nn.Module):
    def __init__(self):
        super().__init__()
        self.denoise_model = EfficientNet_Denoise(mode="b1")
        self.denoise_model.load_state_dict(torch.load(weight_path, map_location="cpu"), strict=True)
        for param in self.denoise_model.parameters():
            param.requires_grad = False
        self.model = EDCNN()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.lin = nn.Linear(1, 1)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        denoise_map = self.denoise_model(x)
        x = x - denoise_map
        out = self.model(x)
        out = self.avg_pool(out)
        out = out.reshape(out.shape[0], -1)
        out = 4 * F.sigmoid(self.lin(out))
        return out


class TwoStage_EDCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.denoise_model = EDCNN()
        self.denoise_model.load_state_dict(torch.load(edcnn_weight_path, map_location="cpu"), strict=True)
        for param in self.denoise_model.parameters():
            param.requires_grad = False
        self.model = EDCNN()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.lin = nn.Linear(1, 1)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        denoise_map = self.denoise_model(x)
        x = x - denoise_map
        out = self.model(x)
        out = self.avg_pool(out)
        out = out.reshape(out.shape[0], -1)
        out = 4 * F.sigmoid(self.lin(out))
        return out

class TwoStage_EDCNNSwin(nn.Module):
    def __init__(self):
        super().__init__()
        self.denoise_model = EDCNN()
        self.denoise_model.load_state_dict(torch.load(edcnn_weight_path, map_location="cpu"), strict=True)
        for param in self.denoise_model.parameters():
            param.requires_grad = False
        self.model = EDCNN_Swin()

        self.relu = nn.LeakyReLU()

    def forward(self, x, mode):
        denoise_map = self.denoise_model(x)
        x = x - denoise_map
        if mode=="train":
            i = random.randint(0, 512 - 96)
            j = random.randint(0, 512 - 96)
            x = x[:, :, i:i + 96, j:j + 96]
        elif mode=="test":
            x = x[:, :, 256-48:256+48,256-48:256+48]
        out = self.model(x)
        return out
