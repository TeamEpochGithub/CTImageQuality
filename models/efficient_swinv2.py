import torch
import torch.nn as nn
from models.efficient_net import EfficientNet_v1
from models.swin_transformer_v2.model_parts import SwinTransformerStage
import torch.nn.functional as F


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

class MixBlock(nn.Module):
    def __init__(self, c_in):
        super(MixBlock, self).__init__()
        self.local_query = nn.Conv2d(c_in, c_in, (1, 1))
        self.global_query = nn.Conv2d(c_in, c_in, (1, 1))

        self.local_key = nn.Conv2d(c_in, c_in, (1, 1))
        self.global_key = nn.Conv2d(c_in, c_in, (1, 1))

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.LeakyReLU(inplace=True)

        self.global_gamma = nn.Parameter(torch.zeros(1))
        self.local_gamma = nn.Parameter(torch.zeros(1))

        self.local_conv = nn.Conv2d(c_in, c_in, (1, 1), groups=c_in)
        self.local_bn = nn.BatchNorm2d(c_in, affine=True)
        self.global_conv = nn.Conv2d(c_in, c_in, (1, 1), groups=c_in)
        self.global_bn = nn.BatchNorm2d(c_in, affine=True)

    def forward(self, x_local, x_global):
        B, C, W, H = x_local.size()
        assert W == H

        q_local = self.local_query(x_local).reshape(-1, W, H)  # [BC, W, H]
        q_global = self.global_query(x_global).reshape(-1, W, H)
        M_query = torch.cat([q_local, q_global], dim=2)  # [BC, W, 2H]

        k_local = self.local_key(x_local).reshape(-1, W, H).transpose(1, 2)  # [BC, H, W]
        k_global = self.global_key(x_global).reshape(-1, W, H).transpose(1, 2)
        M_key = torch.cat([k_local, k_global], dim=1)  # [BC, 2H, W]

        energy = torch.bmm(M_query, M_key)  # [BC, W, W]
        attention = self.softmax(energy).view(B, C, W, W)

        att_global = x_global * attention * (torch.sigmoid(self.global_gamma) * 2.0 - 1.0)
        y_local = x_local + self.relu(self.local_bn(self.local_conv(att_global)))

        att_local = x_local * attention * (torch.sigmoid(self.local_gamma) * 2.0 - 1.0)
        y_global = x_global + self.relu(self.global_bn(self.global_conv(att_local)))
        return y_local, y_global

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


class Efficientnet_Swinv2(nn.Module):
    def __init__(self, configs, hidden_dim=64, layers=(2, 2, 18,
                                                            2), heads=(4, 8, 16, 32), channels=1, head_dim=32,
                 window_size=8, downscaling_factors=(2, 2, 2, 2), relative_pos_embedding=True, out_channel=1):
        super(Efficientnet_Swinv2, self).__init__()
        self.hidden_dim = hidden_dim
        self.efficientnet = EfficientNet_v1(input_dim=32)
        self.out_channel = out_channel
        self.stem = nn.Sequential(
            Conv_3(1, self.hidden_dim, 3, 2, 1),
            Conv_3(self.hidden_dim, self.hidden_dim, 3, 1, 1),
            Conv_3(self.hidden_dim, self.hidden_dim // 2, 3, 1, 1),
        )
        dropout_path = torch.linspace(0., 0.24, 24).tolist()
        self.img_size = configs["img_size"]
        self.use_mix = configs["use_mix"]
        self.use_avg = configs["use_avg"]

        self.stage1 = SwinTransformerStage(
                in_channels=self.hidden_dim // 2,
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
        self.eff1 = self.efficientnet.blocks1
        if self.use_mix:
            self.mix1 = MixBlock(hidden_dim)
        else:
            self.conv11 = Conv_3(hidden_dim * 2, hidden_dim, 3, 1, 1)
            self.conv12 = DConv_5(hidden_dim)

        self.stage2 = SwinTransformerStage(
                in_channels=self.hidden_dim,
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
        self.eff2 = self.efficientnet.blocks2
        if self.use_mix:
            self.mix2 = MixBlock(hidden_dim*2)
        else:
            self.conv21 = Conv_3(hidden_dim * 4, hidden_dim * 2, 3, 1, 1)
            self.conv22 = DConv_5(hidden_dim * 2)

        self.stage3 = SwinTransformerStage(
                in_channels=self.hidden_dim*2,
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
        self.eff3 = self.efficientnet.blocks3
        if self.use_mix:
            self.mix3 = MixBlock(hidden_dim * 4)
        else:
            self.conv31 = Conv_3(hidden_dim * 8, hidden_dim * 4, 3, 1, 1)
            self.conv32 = DConv_5(hidden_dim * 4)

        self.stage4 = SwinTransformerStage(
                in_channels=self.hidden_dim*4,
                depth=layers[3],
                downscale=downscaling_factors[3],
                input_resolution=(self.img_size // 16, self.img_size // 16),
                number_of_heads=heads[3],
                window_size=window_size,
                ff_feature_ratio=4,
                dropout=0.0,
                dropout_attention=0.0,
                dropout_path=dropout_path[22:24],
                use_checkpoint=False,
                sequential_self_attention=False,
                use_deformable_block=False
            )
        self.eff4 = self.efficientnet.blocks4
        if self.use_mix:
            self.mix4 = MixBlock(hidden_dim * 8)

        self.conv41 = Conv_3(hidden_dim * 16, hidden_dim * 8, 3, 1, 1)
        self.conv42 = DConv_5(hidden_dim * 8)

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
        if self.use_mix:
            e0 = self.stem(x)
            e1_swin_tmp = self.stage1(e0)
            e1_res = self.eff1(e0)
            e1_swin_tmp, e1_res = self.mix1(e1_swin_tmp, e1_res)

            e2_swin_tmp = self.stage2(e1_swin_tmp)
            e2_res = self.eff2(e1_res)
            e2_swin_tmp, e2_res = self.mix2(e2_swin_tmp, e2_res)

            e3_swin_tmp = self.stage3(e2_swin_tmp)
            e3_res = self.eff3(e2_res)
            e3_swin_tmp, e3_res = self.mix3(e3_swin_tmp, e3_res)

            e4_swin_tmp = self.stage4(e3_swin_tmp)
            e4_res = self.eff4(e3_res)
            e4_swin_tmp, e4_res = self.mix4(e4_swin_tmp, e4_res)
        else:
            e0 = self.stem(x)  # 64, 128, 128
            e1_swin_tmp = self.stage1(e0)
            e1_res = self.eff1(e0)
            e1 = self.conv11(torch.cat((e1_swin_tmp, e1_res), dim=1))
            e1 = self.conv12(e1)+e1

            e2_swin_tmp = self.stage2(e1)
            e2_res = self.eff2(e1)
            e2 = self.conv21(torch.cat((e2_swin_tmp, e2_res), dim=1))
            e2 = self.conv22(e2)+e2

            e3_swin_tmp = self.stage3(e2)
            e3_res = self.eff3(e2)
            e3 = self.conv31(torch.cat((e3_swin_tmp, e3_res), dim=1))
            e3 = self.conv32(e3)+e3

            e4_swin_tmp = self.stage4(e3)
            e4_res = self.eff4(e3)

        e4 = torch.cat((e4_swin_tmp, e4_res), dim=1)
        e4 = self.conv41(e4)
        e4 = self.conv42(e4) + e4

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
# model = Efficient_Swinv2_Next()
# ous = model(ins)
# print(ous.shape)
