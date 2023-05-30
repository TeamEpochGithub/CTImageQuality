import torch
import torch.nn as nn
from models.swin import StageModule
import torchvision
import torch.nn.functional as F
from models.swin_transformer_v2.model_parts import SwinTransformerStage

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
            nn.Conv2d(channels, channels,3, 1, padding=2, dilation=2),
            nn.BatchNorm2d(channels, affine=True),
            nn.LeakyReLU(inplace=True)
        )
        self.layer3 = Conv_3(channels, channels, 3, 1, 1)

    def forward(self, x):
        e1 = self.layer1(x)
        e2 = self.layer2(e1)
        e2 = e2+x
        e3 = self.layer3(e2)
        e3 = e3+e1
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
        e2=e2+x
        return e2

# MCNN (5 layers)
class DConv_5(nn.Module):
    def __init__(self, channels, alpha=0.2):
        super().__init__()
        self.layer1 = Conv_3(channels, channels, 3, 1, 1)
        self.layer2 = nn.Sequential(
            nn.Conv2d(channels, channels,3, 1, padding=2, dilation=2),
            nn.BatchNorm2d(channels, affine=True),
            nn.LeakyReLU(inplace=True)
        )
        self.layer3 = Conv_3(channels, channels, 3, 1, 1)
        self.layer4 = nn.Sequential(
            nn.Conv2d(channels, channels,3, 1, padding=4, dilation=4),
            nn.BatchNorm2d(channels, affine=True),
            nn.LeakyReLU(inplace=True)
        )
        self.layer5 = Conv_3(channels, channels, 3, 1, 1)

    def forward(self, x):
        e1 = self.layer1(x)
        e2 = self.layer2(e1)
        e2 = e2+x
        e3 = self.layer3(e2)
        e3 = e3+e1
        e4 = self.layer4(e3)
        e4 = e4+e2
        e5 = self.layer5(e4)
        e5 = e5+e3
        return e5

# UNet34-Swin
class Resnet34_Swinv2(nn.Module):
    def __init__(self, configs, hidden_dim=64, layers=(2, 2, 18,
                                                            2), heads=(4, 8, 16, 32), channels=1, head_dim=32,
                 window_size=8, downscaling_factors=(2, 2, 2, 2), relative_pos_embedding=True, out_channel=1):
        super(Resnet34_Swinv2, self).__init__()
        self.base_model = torchvision.models.resnet34(True)
        self.base_layers = list(self.base_model.children())

        self.img_size = configs["img_size"]
        self.use_avg = configs["use_avg"]
        self.out_channel = out_channel

        self.layer0 = nn.Sequential(
            Conv_3(channels, hidden_dim//2, 3, 2, 1),
            Conv_3(hidden_dim//2, hidden_dim//2, 3, 1, 1),
            Conv_3(hidden_dim//2, hidden_dim//2, 3, 1, 1),
        )

        dropout_path = torch.linspace(0., 0.24, 24).tolist()
        self.stage1 = SwinTransformerStage(
                in_channels=hidden_dim//2,
                depth=layers[0],
                downscale=downscaling_factors[0],
                input_resolution=(self.img_size // 2, self.img_size // 2),
                number_of_heads=heads[0],
                window_size=window_size,
                ff_feature_ratio=4,
                dropout=0.0,
                dropout_attention=0.0,
                dropout_path=dropout_path[0:2],
                use_checkpoint=False,
                sequential_self_attention=False,
                use_deformable_block=False
            )

        self.avg1 = Channel_wise(hidden_dim // 2, hidden_dim, [hidden_dim, self.img_size // 4, self.img_size // 4])

        self.res_convs1 = nn.Sequential(*self.base_layers[4])

        self.stage2 = SwinTransformerStage(
                in_channels=hidden_dim,
                depth=layers[1],
                downscale=downscaling_factors[1],
                input_resolution=(self.img_size // 4, self.img_size // 4),
                number_of_heads=heads[1],
                window_size=window_size,
                ff_feature_ratio=4,
                dropout=0.0,
                dropout_attention=0.0,
                dropout_path=dropout_path[2:4],
                use_checkpoint=False,
                sequential_self_attention=False,
                use_deformable_block=False
            )

        self.avg2 = Channel_wise(hidden_dim, hidden_dim * 2, [hidden_dim * 2, self.img_size // 8, self.img_size // 8])

        self.res_convs2 = nn.Sequential(*(self.base_layers[5][1:]))

        self.avg3 = Channel_wise(hidden_dim * 2, hidden_dim * 4, [hidden_dim * 4, self.img_size // 16, self.img_size // 16])

        self.stage3 = SwinTransformerStage(
                in_channels=hidden_dim*2,
                depth=layers[2],
                downscale=downscaling_factors[2],
                input_resolution=(self.img_size // 8, self.img_size // 8),
                number_of_heads=heads[2],
                window_size=window_size,
                ff_feature_ratio=4,
                dropout=0.0,
                dropout_attention=0.0,
                dropout_path=dropout_path[4:22],
                use_checkpoint=False,
                sequential_self_attention=False,
                use_deformable_block=False
            )

        self.res_convs3 = nn.Sequential(*(self.base_layers[6][1:]))

        self.avg4 = Channel_wise(hidden_dim * 4, hidden_dim * 8, [hidden_dim * 8, self.img_size // 32, self.img_size // 32])

        self.stage4 = SwinTransformerStage(
                in_channels=hidden_dim*4,
                depth=layers[3],
                downscale=downscaling_factors[3],
                input_resolution=(self.img_size // 16, self.img_size // 16),
                number_of_heads=heads[3],
                window_size=window_size,
                ff_feature_ratio=4,
                dropout=0.0,
                dropout_attention=0.0,
                dropout_path=dropout_path[22:],
                use_checkpoint=False,
                sequential_self_attention=False,
                use_deformable_block=False
            )

        self.res_convs4 = nn.Sequential(*(self.base_layers[7][1:]))

        self.out_conv1 = Conv_3(512, 512, 3, 2, 1)
        self.out_conv2 = DConv_5(512)
        self.out_conv3 = Conv_3(512, 512, 3, 2, 1)
        self.out_conv4 = DConv_5(512)

        if self.use_avg:
            f_size = self.img_size // 128
            self.avg_pool = nn.AvgPool2d(f_size)
        else:
            f_size = 512 * (self.img_size // 128) ** 2
            self.fc1 = nn.Linear(f_size, 512)
            self.l_relu = nn.LeakyReLU(inplace=True)
        self.fc2 = nn.Linear(512, out_channel)



    def forward(self, x):
        e0 = self.layer0(x)
        e1_swin_tmp = self.stage1(e0) + self.avg1(e0)
        e1 = self.res_convs1(e1_swin_tmp)+e1_swin_tmp

        e2_swin_tmp = self.stage2(e1) + self.avg2(e1)
        e2 = self.res_convs2(e2_swin_tmp)+e2_swin_tmp

        e3_swin_tmp = self.stage3(e2) + self.avg3(e2)
        e3 = self.res_convs3(e3_swin_tmp)+e3_swin_tmp

        e4_swin_tmp = self.stage4(e3) + self.avg4(e3)
        e4 = self.res_convs4(e4_swin_tmp)+e4_swin_tmp

        e4 = self.out_conv1(e4)
        e4 = self.out_conv2(e4)+e4
        e4 = self.out_conv3(e4)
        outs = self.out_conv4(e4)+e4

        if self.use_avg:
            outs = self.avg_pool(outs)
            outs = outs.reshape(outs.shape[0], -1)
        else:
            outs = outs.reshape(outs.shape[0], -1)
            outs = self.l_relu(self.fc1(outs))
        if self.out_channel==1:
            outs = 4 * F.sigmoid(self.fc2(outs))
        else:
            outs = self.fc2(outs)
        return outs

# ins = torch.rand((8, 1, 256, 256))
# model = Resnet34_Swinv2()
# ous = model(ins)
# print(ous.shape)
