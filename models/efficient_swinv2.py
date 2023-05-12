import torch
import torch.nn as nn
from models.efficient_net import EfficientNet_v1
from models.swin_transformer_v2.model_parts import SwinTransformerStage
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.conv_relu(x1)
        return x1


class Convs(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Convs, self).__init__()
        self.conv_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_relu(x)


class Decoder_v1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder_v1, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        x = self.conv_relu(x)
        return x

class Conv_3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding, alpha=0.2):
        super(Conv_3, self).__init__()
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv3(x)

class Efficientnet_Swinv2(nn.Module):
    def __init__(self, size=256):
        super(Efficientnet_Swinv2, self).__init__()
        self.input_size = size
        self.efficientnet = EfficientNet_v1()
        self.stem = nn.Sequential(
            Conv_3(1, 64, 7, 2, 3),
            Conv_3(64, 64, 3, 1, 1),
            Conv_3(64, 64, 3, 1, 1),
        )
        dropout_path = torch.linspace(0., 0.2, 8).tolist()
        self.stage1 = SwinTransformerStage(
                in_channels=64,
                depth=2,
                downscale=2,
                input_resolution=(self.input_size //2, self.input_size // 2),
                number_of_heads=4,
                window_size=8,
                ff_feature_ratio=4,
                dropout=0.0,
                dropout_attention=0.0,
                dropout_path=dropout_path[0:2],
                use_checkpoint=False,
                sequential_self_attention=False,
                use_deformable_block=False
            )
        self.eff1 = self.efficientnet.blocks1

        self.stage2 = SwinTransformerStage(
                in_channels=128,
                depth=2,
                downscale=2,
                input_resolution=(self.input_size // 4, self.input_size // 4),
                number_of_heads=8,
                window_size=8,
                ff_feature_ratio=4,
                dropout=0.0,
                dropout_attention=0.0,
                dropout_path=dropout_path[2:4],
                use_checkpoint=False,
                sequential_self_attention=False,
                use_deformable_block=False
            )
        self.eff2 = self.efficientnet.blocks2

        self.stage3 = SwinTransformerStage(
                in_channels=256,
                depth=2,
                downscale=2,
                input_resolution=(self.input_size // 8, self.input_size // 8),
                number_of_heads=16,
                window_size=8,
                ff_feature_ratio=4,
                dropout=0.0,
                dropout_attention=0.0,
                dropout_path=dropout_path[4:6],
                use_checkpoint=False,
                sequential_self_attention=False,
                use_deformable_block=False
            )
        self.eff3 = self.efficientnet.blocks3

        self.stage4 = SwinTransformerStage(
                in_channels=512,
                depth=2,
                downscale=2,
                input_resolution=(self.input_size // 16, self.input_size // 16),
                number_of_heads=32,
                window_size=4,
                ff_feature_ratio=4,
                dropout=0.0,
                dropout_attention=0.0,
                dropout_path=dropout_path[6:8],
                use_checkpoint=False,
                sequential_self_attention=False,
                use_deformable_block=False
            )
        self.eff4 = self.efficientnet.blocks4

        self.avg = nn.AvgPool2d(8)
        self.fc = nn.Linear(1024, 1)

    def forward(self, x):
        x1 = self.stem(x)  # 64, 128, 128
        x1 = self.stage1(x1)
        x2 = self.eff1(x1)+x1  # 96, 128, 128
        x2 = self.stage2(x2)  # 192, 64, 64
        x3 = self.eff2(x2)+x2
        x3 = self.stage3(x3)  # 384, 32, 32
        x4 = self.eff3(x3)+x3
        x4 = self.stage4(x4)  # 192, 16, 16
        x5 = self.eff4(x4)+x4

        outs = self.avg(x5)
        outs = outs.view(outs.shape[0], -1)
        outs = F.relu(self.fc(outs))

        return outs

# ins = torch.rand((8, 1, 256, 256))
# model = Efficient_Swinv2_Next()
# ous = model(ins)
# print(ous.shape)
