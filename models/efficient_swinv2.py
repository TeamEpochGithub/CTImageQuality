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
    def __init__(self, size=512, hidden_dim = 64):
        super(Efficientnet_Swinv2, self).__init__()
        self.input_size = size
        self.hidden_dim = hidden_dim
        self.efficientnet = EfficientNet_v1(input_dim=32)
        self.stem = nn.Sequential(
            Conv_3(1, self.hidden_dim, 7, 2, 3),
            Conv_3(self.hidden_dim, self.hidden_dim, 3, 1, 1),
            Conv_3(self.hidden_dim, self.hidden_dim // 2, 3, 1, 1),
        )
        dropout_path = torch.linspace(0., 0.2, 8).tolist()
        self.stage1 = SwinTransformerStage(
                in_channels=self.hidden_dim // 2,
                depth=2,
                downscale=2,
                input_resolution=(self.input_size // 2, self.input_size // 2),
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
        self.conv11 = nn.Sequential(
            Conv_3(hidden_dim * 2, hidden_dim, 3, 1, 1),
            Conv_3(hidden_dim, hidden_dim, 3, 1, 1),
        )

        self.stage2 = SwinTransformerStage(
                in_channels=self.hidden_dim,
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
        self.conv12 = nn.Sequential(
            Conv_3(hidden_dim * 4, hidden_dim * 2, 3, 1, 1),
            Conv_3(hidden_dim * 2, hidden_dim * 2, 3, 1, 1),
        )


        self.stage3 = SwinTransformerStage(
                in_channels=self.hidden_dim*2,
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
        self.conv13 = nn.Sequential(
            Conv_3(hidden_dim * 8, hidden_dim * 4, 3, 1, 1),
            Conv_3(hidden_dim * 4, hidden_dim * 4, 3, 1, 1),
        )

        self.stage4 = SwinTransformerStage(
                in_channels=self.hidden_dim*4,
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
        self.conv14 = nn.Sequential(
            Conv_3(hidden_dim * 16, hidden_dim * 8, 3, 1, 1),
            Conv_3(hidden_dim * 8, hidden_dim * 8, 3, 1, 1),
        )

        self.avg = nn.AvgPool2d(16)
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        e0 = self.stem(x)  # 64, 128, 128
        e1_swin_tmp = self.stage1(e0)
        e1_res = self.eff1(e0)
        e1 = self.conv11(torch.cat((e1_swin_tmp, e1_res), dim=1))

        e2_swin_tmp = self.stage2(e1)
        e2_res = self.eff2(e1)
        e2 = self.conv12(torch.cat((e2_swin_tmp, e2_res), dim=1))

        e3_swin_tmp = self.stage3(e2)
        e3_res = self.eff3(e2)
        e3 = self.conv13(torch.cat((e3_swin_tmp, e3_res), dim=1))

        e4_swin_tmp = self.stage4(e3)
        e4_res = self.eff4(e3)
        e4 = self.conv14(torch.cat((e4_swin_tmp, e4_res), dim=1))

        outs = self.avg(e4)
        outs = outs.view(outs.shape[0], -1)
        outs = F.relu(self.fc(outs))

        return outs

# ins = torch.rand((8, 1, 256, 256))
# model = Efficient_Swinv2_Next()
# ous = model(ins)
# print(ous.shape)
