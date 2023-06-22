import torch
import torch.nn as nn
import sys
sys.path.append('..')
from pretrain.pretrain_models.efficientnet_unet import EfficientNet_Denoise
import torchvision.models as models
import torch.nn.functional as F

weight_path = r"C:\Users\leo\Documents\CTImageQuality\pretrain\weights\Efficientnet_B1\pretrain_weight_denoise.pkl"

class TwoStage_EfficientB0(nn.Module):
    def __init__(self):
        super().__init__()
        self.denoise_model = EfficientNet_Denoise(mode="b1")
        self.denoise_model.load_state_dict(torch.load(weight_path, map_location="cpu"), strict=True)
        for param in self.denoise_model.parameters():
            param.requires_grad = False
        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, 1)

    def forward(self, x):
        noise_x = self.denoise_model(x)
        outs = self.resnet18(noise_x)
        outs = 4*F.sigmoid(outs)
        return outs

# ins = torch.randn(8, 1, 512, 512)
# model = TwoStage_EfficientB0()
# outs = model(ins)
# print(outs)
