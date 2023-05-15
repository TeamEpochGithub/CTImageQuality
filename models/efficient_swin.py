import torch
import torch.nn as nn
from models.swin import StageModule
import torchvision
import torch.nn.functional as F
from models.efficient_net import EfficientNet_v1

# Conv Block: Conv+BatchNorm+ReLU
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

# DConv Block: DConv+BatchNorm+ReLU
class DConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding, dilation, alpha=0.2):
        super(DConv, self).__init__()
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True)
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
            nn.ReLU(inplace=True)
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
            nn.Conv2d(out_channels, out_channels, 1),
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
            nn.ReLU(inplace=True)
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
            nn.ReLU(inplace=True)
        )
        self.layer3 = Conv_3(channels, channels, 3, 1, 1)
        self.layer4 = nn.Sequential(
            nn.Conv2d(channels, channels,3, 1, padding=4, dilation=4),
            nn.BatchNorm2d(channels, affine=True),
            nn.ReLU(inplace=True)
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

class Efficientnet_Swin(nn.Module):
    def __init__(self, img_size=512, hidden_dim=64, layers=(2, 2, 18,
                                                            2), heads=(3, 6, 12, 24), channels=1, head_dim=32,
                 window_size=8, downscaling_factors=(2, 2, 2, 2), relative_pos_embedding=True):
        super(Efficientnet_Swin, self).__init__()
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

        self.efficient_model = EfficientNet_v1(input_dim=64)
        self.res_convs1 = self.efficient_model.blocks1
        self.conv11 = nn.Sequential(
            Conv_3(hidden_dim * 2, hidden_dim, 3, 1, 1),
            Conv_3(hidden_dim, hidden_dim, 3, 1, 1),
        )

        self.stage2 = StageModule(in_channels=hidden_dim, hidden_dimension=hidden_dim * 2, layers=layers[1],
                                  downscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)


        self.res_convs2 = self.efficient_model.blocks2
        self.conv12 = nn.Sequential(
            Conv_3(hidden_dim * 4, hidden_dim * 2, 3, 1, 1),
            Conv_3(hidden_dim * 2, hidden_dim * 2, 3, 1, 1),
        )


        self.stage3 = StageModule(in_channels=hidden_dim * 2, hidden_dimension=hidden_dim * 4, layers=layers[2],
                                  downscaling_factor=downscaling_factors[2], num_heads=heads[2], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)

        self.res_convs3 = self.efficient_model.blocks3
        self.conv13 = nn.Sequential(
            Conv_3(hidden_dim * 8, hidden_dim * 4, 3, 1, 1),
            Conv_3(hidden_dim * 4, hidden_dim * 4, 3, 1, 1),
        )


        self.stage4 = StageModule(in_channels=hidden_dim * 4, hidden_dimension=hidden_dim * 8, layers=layers[3],
                                  downscaling_factor=downscaling_factors[3], num_heads=heads[3], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)

        self.res_convs4 = self.efficient_model.blocks4
        self.conv14 = nn.Sequential(
            Conv_3(hidden_dim * 16, hidden_dim * 8, 3, 1, 1),
            Conv_3(hidden_dim * 8, hidden_dim * 8, 3, 1, 1),
        )

        self.avg = nn.AvgPool2d(16)
        self.fc = nn.Linear(512, 1)



    def forward(self, x):
        e0 = self.layer0(x)
        e1_swin_tmp = self.stage1(e0)
        e1_res = self.res_convs1(e0)
        e1 = self.conv11(torch.cat((e1_swin_tmp, e1_res), dim=1))

        e2_swin_tmp = self.stage2(e1)
        e2_res = self.res_convs2(e1)
        e2 = self.conv12(torch.cat((e2_swin_tmp, e2_res), dim=1))

        e3_swin_tmp = self.stage3(e2)
        e3_res = self.res_convs3(e2)
        e3 = self.conv13(torch.cat((e3_swin_tmp, e3_res), dim=1))

        e4_swin_tmp = self.stage4(e3)
        e4_res = self.res_convs4(e3)
        e4 = self.conv14(torch.cat((e4_swin_tmp, e4_res), dim=1))

        outs = self.avg(e4)
        outs = outs.view(outs.shape[0], -1)
        outs = F.relu(self.fc(outs))
        return outs

# ins = torch.rand((8, 1, 256, 256))
# model = Efficientnet_Swin()
# ous = model(ins)
# print(ous.shape)
