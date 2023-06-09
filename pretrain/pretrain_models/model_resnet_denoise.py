import torch
import torch.nn as nn
import sys

import torchvision

from pretrain.pretrain_models.swin import StageModule


# Conv Block: Conv+BatchNorm+ReLU
class Conv_3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding, alpha=0.2):
        super(Conv_3, self).__init__()
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv3(x)


# DConv Block: DConv+BatchNorm+ReLU
class DConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding, dilation, alpha=0.2):
        super(DConv, self).__init__()
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv3(x)


# Single Decoder Block
class Decoder(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, alpha=0.2):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.conv_relu(x1)
        return x1


# Auxiliary branch of swin transformer module, conv + layernorm
class Channel_wise(nn.Module):
    def __init__(self, in_channels, out_channels, sizes):
        super().__init__()
        self.avg = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 2, 2),
            DConv_5(out_channels),
            nn.LayerNorm(sizes)
        )

    def forward(self, x):
        return self.avg(x)


# MCNN (3 layers)
class DConv_3(nn.Module):
    def __init__(self, channels, alpha=0.2):
        super().__init__()
        self.layer1 = Conv_3(channels, channels, 3, 1, 1)
        self.layer2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, padding=2, dilation=2),
            nn.BatchNorm2d(channels, affine=True),
            nn.LeakyReLU(inplace=True)
        )
        self.layer3 = Conv_3(channels, channels, 3, 1, 1)

    def forward(self, x):
        e1 = self.layer1(x)
        e2 = self.layer2(e1)
        e2 = e2 + x
        e3 = self.layer3(e2)
        e3 = e3 + e1
        return e3


# Conv*2
class DConv_2(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layer1 = Conv_3(channels, channels, 3, 1, 1)
        self.layer2 = Conv_3(channels, channels, 3, 1, 1)

    def forward(self, x):
        e1 = self.layer1(x)
        e2 = self.layer2(e1)
        e2 = e2 + x
        return e2


# MCNN (5 layers)
class DConv_5(nn.Module):
    def __init__(self, channels, alpha=0.2):
        super().__init__()
        self.layer1 = Conv_3(channels, channels, 3, 1, 1)
        self.layer2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, padding=2, dilation=2),
            nn.BatchNorm2d(channels, affine=True),
            nn.LeakyReLU(inplace=True)
        )
        self.layer3 = Conv_3(channels, channels, 3, 1, 1)
        self.layer4 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, padding=4, dilation=4),
            nn.BatchNorm2d(channels, affine=True),
            nn.LeakyReLU(inplace=True)
        )
        self.layer5 = Conv_3(channels, channels, 3, 1, 1)

    def forward(self, x):
        e1 = self.layer1(x)
        e2 = self.layer2(e1)
        e2 = e2 + x
        e3 = self.layer3(e2)
        e3 = e3 + e1
        e4 = self.layer4(e3)
        e4 = e4 + e2
        e5 = self.layer5(e4)
        e5 = e5 + e3
        return e5


class Resnet34_Swin_Denoise(nn.Module):
    def __init__(self, img_size=512, hidden_dim=64, layers=(2, 2, 18,
                                                            2), heads=(3, 6, 12, 24), channels=1, head_dim=32,
                 window_size=8, downscaling_factors=(2, 2, 2, 2), relative_pos_embedding=True):
        super(Resnet34_Swin_Denoise, self).__init__()
        self.base_model = torchvision.models.resnet34(True)
        self.base_layers = list(self.base_model.children())
        self.layer0 = nn.Sequential(
            Conv_3(channels, hidden_dim, 7, 2, 3),
            Conv_3(hidden_dim, hidden_dim, 3, 1, 1),
            Conv_3(hidden_dim, hidden_dim, 3, 1, 1),
        )

        self.stage1 = StageModule(in_channels=hidden_dim, hidden_dimension=hidden_dim, layers=layers[0],
                                  downscaling_factor=downscaling_factors[0], num_heads=heads[0], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)

        self.avg1 = Channel_wise(hidden_dim, hidden_dim, [hidden_dim, img_size // 4, img_size // 4])

        self.res_convs1 = nn.Sequential(*self.base_layers[4])

        self.stage2 = StageModule(in_channels=hidden_dim, hidden_dimension=hidden_dim * 2, layers=layers[1],
                                  downscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)

        self.avg2 = Channel_wise(hidden_dim, hidden_dim * 2, [hidden_dim * 2, img_size // 8, img_size // 8])

        self.res_convs2 = nn.Sequential(*(self.base_layers[5][1:]))

        self.avg3 = Channel_wise(hidden_dim * 2, hidden_dim * 4, [hidden_dim * 4, img_size // 16, img_size // 16])

        self.stage3 = StageModule(in_channels=hidden_dim * 2, hidden_dimension=hidden_dim * 4, layers=layers[2],
                                  downscaling_factor=downscaling_factors[2], num_heads=heads[2], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)

        self.res_convs3 = nn.Sequential(*(self.base_layers[6][1:]))

        self.avg4 = Channel_wise(hidden_dim * 4, hidden_dim * 8, [hidden_dim * 8, img_size // 32, img_size // 32])

        self.stage4 = StageModule(in_channels=hidden_dim * 4, hidden_dimension=hidden_dim * 8, layers=layers[3],
                                  downscaling_factor=downscaling_factors[3], num_heads=heads[3], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)

        self.res_convs4 = nn.Sequential(*(self.base_layers[7][1:]))

        self.decode4 = Decoder(512, 256 + 256, 256)
        self.decode3 = Decoder(256, 128 + 128, 128)
        self.decode2 = Decoder(128, 64 + 64, 64)
        self.decode1 = Decoder(64, 64 + 64, 64)
        self.decode0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_last = nn.Conv2d(64, channels, 1)

    def forward(self, x):
        e0 = self.layer0(x)
        e1_swin_tmp = self.stage1(e0) + self.avg1(e0)
        e1 = self.res_convs1(e1_swin_tmp) + e1_swin_tmp

        e2_swin_tmp = self.stage2(e1) + self.avg2(e1)
        e2 = self.res_convs2(e2_swin_tmp) + e2_swin_tmp

        e3_swin_tmp = self.stage3(e2) + self.avg3(e2)
        e3 = self.res_convs3(e3_swin_tmp) + e3_swin_tmp

        e4_swin_tmp = self.stage4(e3) + self.avg4(e3)
        e4 = self.res_convs4(e4_swin_tmp) + e4_swin_tmp

        d4 = self.decode4(e4, e3)  # 256,16,16
        d3 = self.decode3(d4, e2)  # 256,32,32
        d2 = self.decode2(d3, e1)  # 128,64,64
        d1 = self.decode1(d2, e0)  # 64,128,128
        d0 = self.decode0(d1)  # 64,256,256
        out = self.conv_last(d0)  # 1,256,256
        return out

# ins = torch.rand(8, 1, 512, 512)
# model = Resnet34_Swin()
# print(model(ins).shape)
