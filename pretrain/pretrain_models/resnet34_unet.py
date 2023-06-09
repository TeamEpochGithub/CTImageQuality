import torch
import torch.nn as nn
from torchvision import models


class UNet34_Denoise(nn.Module):
    def __init__(self, n_class=1):
        super().__init__()

        self.base_model = models.resnet34(pretrained=True)
        self.base_layers = list(self.base_model.children())

        # self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        self.layer0_1x1 = self.conv1x1(64, 64)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = self.conv1x1(64, 64)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = self.conv1x1(128, 128)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = self.conv1x1(256, 256)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = self.conv1x1(512, 512)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = self.conv1x1(256 + 512, 512)
        self.conv_up2 = self.conv1x1(128 + 512, 256)
        self.conv_up1 = self.conv1x1(64 + 256, 256)
        self.conv_up0 = self.conv1x1(64 + 256, 128)

        self.conv_original_size0 = self.conv1x1(1, 64)
        self.conv_original_size1 = self.conv1x1(64, 64)
        self.conv_original_size2 = self.conv1x1(64 + 128, 64)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out

    def conv1x1(self, in_channels, out_channels, groups=1):
        return nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, stride=1)

