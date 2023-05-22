import torch
import torch.nn as nn
from swin import StageModule
import torchvision
from efficient_net import EfficientNet_v1


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


class Channel_wise(nn.Module):
    def __init__(self, in_channels, out_channels, sizes):
        super().__init__()
        self.avg = nn.Sequential(
            Conv_3(in_channels, out_channels, 2, 2, 0),
            Conv_3(out_channels, out_channels, 1, 1, 0)
            # nn.Conv2d(in_channels, out_channels, 2, 2),
            # nn.Conv2d(out_channels, out_channels, 1),
            # nn.LayerNorm(sizes)
        )

    def forward(self, x):
        return self.avg(x)


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


class DConv_2(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layer1 = DConv(channels, channels, 3, 1, 2, dilation=2)
        self.layer2 = DConv(channels, channels, 3, 1, 4, dilation=4)

    def forward(self, x):
        e1 = self.layer1(x)
        e2 = self.layer2(e1)
        e2 = e2 + x
        return e2


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

# Mix Block with attention mechanism
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


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


def _BatchNorm(channels, eps=1e-3, momentum=0.01):
    return nn.BatchNorm2d(channels, eps=eps, momentum=momentum)


def _Conv3x3Bn(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
        _BatchNorm(out_channels),
        Swish()
    )


class Efficient_Swin(nn.Module):
    def __init__(self, hidden_dim=64, layers=(2, 2, 18, 2), heads=(4, 8, 16, 32), channels=1, head_dim=32,
                 window_size=8, downscaling_factors=(2, 2, 2, 2), relative_pos_embedding=True):
        super(Efficient_Swin, self).__init__()

        self.layer0 = nn.Sequential(
            Conv_3(channels, hidden_dim, 3, 2, 1),
            Conv_3(hidden_dim, hidden_dim, 3, 1, 1),
            Conv_3(hidden_dim, hidden_dim, 3, 1, 1),
        )

        self.stage1 = StageModule(in_channels=hidden_dim, hidden_dimension=hidden_dim, layers=layers[0],
                                  downscaling_factor=downscaling_factors[0], num_heads=heads[0], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)

        self.efficient_model = EfficientNet_v1(input_dim=64)

        self.res_convs1 = self.efficient_model.blocks1


        self.mix1 = MixBlock(hidden_dim)
        self.conv11 = Conv_3(hidden_dim * 2, hidden_dim, 3, 1, 1)
        self.conv12 = DConv_5(hidden_dim)

        self.conv1 = Conv_3(hidden_dim * 2, hidden_dim, 3, 1, 1)

        self.stage2 = StageModule(in_channels=hidden_dim, hidden_dimension=hidden_dim * 2, layers=layers[1],
                                  downscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)

        self.res_convs2 = self.efficient_model.blocks2

        self.mix2 = MixBlock(hidden_dim*2)
        self.conv21 = Conv_3(hidden_dim * 4, hidden_dim * 2, 3, 1, 1)
        self.conv22 = DConv_5(hidden_dim * 2)

        self.stage3 = StageModule(in_channels=hidden_dim * 2, hidden_dimension=hidden_dim * 4, layers=layers[2],
                                  downscaling_factor=downscaling_factors[2], num_heads=heads[2], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)

        self.res_convs3 = self.efficient_model.blocks3

        self.mix3 = MixBlock(hidden_dim * 4)
        self.conv31 = Conv_3(hidden_dim * 8, hidden_dim * 4, 3, 1, 1)
        self.conv32 = DConv_5(hidden_dim * 4)

        self.stage4 = StageModule(in_channels=hidden_dim * 4, hidden_dimension=hidden_dim * 8, layers=layers[3],
                                  downscaling_factor=downscaling_factors[3], num_heads=heads[3], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)

        self.res_convs4 = self.efficient_model.blocks4

        self.mix4 = MixBlock(hidden_dim * 8)
        self.conv41 = Conv_3(hidden_dim * 16, hidden_dim * 8, 3, 1, 1)
        self.conv42 = DConv_5(hidden_dim * 8)

        self.decode4 = Decoder(512, 256 + 256, 256)
        self.decode3 = Decoder(256, 128 + 128, 128)
        self.decode2 = Decoder(128, 64 + 64, 64)
        self.decode1 = Decoder(64, 64 + 64, 64)
        self.decode0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
        )
        self.conv_last = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        e0 = self.layer0(x)

        e1_res = self.res_convs1(e0)
        e1_swin = self.stage1(e0)
        e1_res, e1_swin = self.mix1(e1_res, e1_swin)
        e1 = self.conv11(torch.cat((e1_swin, e1_res), dim=1))
        e1 = self.conv12(e1) + e1

        e2_res = self.res_convs2(e1_res)
        e2_swin = self.stage2(e1_swin)
        e2_res, e2_swin = self.mix2(e2_res, e2_swin)
        e2 = self.conv21(torch.cat((e2_swin, e2_res), dim=1))
        e2 = self.conv22(e2) + e2

        e3_res = self.res_convs3(e2_res)
        e3_swin = self.stage3(e2_swin)
        e3_res, e3_swin = self.mix3(e3_res, e3_swin)
        e3 = self.conv31(torch.cat((e3_swin, e3_res), dim=1))
        e3 = self.conv32(e3) + e3

        e4_res = self.res_convs4(e3_res)
        e4_swin = self.stage4(e3_swin)
        e4_res, e4_swin = self.mix4(e4_res, e4_swin)
        e4 = torch.cat((e4_swin, e4_res), dim=1)
        e4 = self.conv41(e4)
        e4 = self.conv42(e4) + e4

        d4 = self.decode4(e4, e3)  # 256,16,16
        d3 = self.decode3(d4, e2)  # 256,32,32
        d2 = self.decode2(d3, e1)  # 128,64,64
        d1 = self.decode1(d2, e0)  # 64,128,128
        d0 = self.decode0(d1)  # 64,256,256
        out = self.conv_last(d0)  # 1,256,256
        return out

ins = torch.rand(8, 1, 512, 512)
model = Efficient_Swin()
ous = model(ins)
print(ous.shape)