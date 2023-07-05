import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torchvision import models


class SobelConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, requires_grad=True):
        assert kernel_size % 2 == 1, 'SobelConv2d\'s kernel_size must be odd.'
        assert out_channels % 4 == 0, 'SobelConv2d\'s out_channels must be a multiple of 4.'
        assert out_channels % groups == 0, 'SobelConv2d\'s out_channels must be a multiple of groups.'

        super(SobelConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # In non-trainable case, it turns into normal Sobel operator with fixed weight and no bias.
        self.bias = bias if requires_grad else False

        if self.bias:
            self.bias = nn.Parameter(torch.zeros(size=(out_channels,), dtype=torch.float32), requires_grad=True)
        else:
            self.bias = None

        self.sobel_weight = nn.Parameter(torch.zeros(
            size=(out_channels, int(in_channels / groups), kernel_size, kernel_size)), requires_grad=False)

        # Initialize the Sobel kernal
        kernel_mid = kernel_size // 2
        for idx in range(out_channels):
            if idx % 4 == 0:
                self.sobel_weight[idx, :, 0, :] = -1
                self.sobel_weight[idx, :, 0, kernel_mid] = -2
                self.sobel_weight[idx, :, -1, :] = 1
                self.sobel_weight[idx, :, -1, kernel_mid] = 2
            elif idx % 4 == 1:
                self.sobel_weight[idx, :, :, 0] = -1
                self.sobel_weight[idx, :, kernel_mid, 0] = -2
                self.sobel_weight[idx, :, :, -1] = 1
                self.sobel_weight[idx, :, kernel_mid, -1] = 2
            elif idx % 4 == 2:
                self.sobel_weight[idx, :, 0, 0] = -2
                for i in range(0, kernel_mid + 1):
                    self.sobel_weight[idx, :, kernel_mid - i, i] = -1
                    self.sobel_weight[idx, :, kernel_size - 1 - i, kernel_mid + i] = 1
                self.sobel_weight[idx, :, -1, -1] = 2
            else:
                self.sobel_weight[idx, :, -1, 0] = -2
                for i in range(0, kernel_mid + 1):
                    self.sobel_weight[idx, :, kernel_mid + i, i] = -1
                    self.sobel_weight[idx, :, i, kernel_mid + i] = 1
                self.sobel_weight[idx, :, 0, -1] = 2

        # Define the trainable sobel factor
        if requires_grad:
            self.sobel_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32),
                                             requires_grad=True)
        else:
            self.sobel_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32),
                                             requires_grad=False)

    def forward(self, x):
        if torch.cuda.is_available():
            self.sobel_factor = self.sobel_factor.cuda()
            if isinstance(self.bias, nn.Parameter):
                self.bias = self.bias.cuda()

        sobel_weight = self.sobel_weight * self.sobel_factor

        if torch.cuda.is_available():
            sobel_weight = sobel_weight.cuda()

        out = F.conv2d(x, sobel_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        return out


class EDCNN(nn.Module):

    def __init__(self, in_ch=1, out_ch=32, sobel_ch=32):
        super(EDCNN, self).__init__()

        self.conv_sobel = SobelConv2d(in_ch, sobel_ch, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv_p1 = nn.Conv2d(in_ch + sobel_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f1 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.conv_p2 = nn.Conv2d(in_ch + sobel_ch + out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.conv_p3 = nn.Conv2d(in_ch + sobel_ch + out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f3 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.conv_p4 = nn.Conv2d(in_ch + sobel_ch + out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f4 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.conv_p5 = nn.Conv2d(in_ch + sobel_ch + out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f5 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.conv_p6 = nn.Conv2d(in_ch + sobel_ch + out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f6 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.conv_p7 = nn.Conv2d(in_ch + sobel_ch + out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f7 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.conv_p8 = nn.Conv2d(in_ch + sobel_ch + out_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_f8 = nn.Conv2d(out_ch, in_ch, kernel_size=3, stride=1, padding=1)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out_0 = self.conv_sobel(x)
        out_0 = torch.cat((x, out_0), dim=-3)

        out_1 = self.relu(self.conv_p1(out_0))
        out_1 = self.relu(self.conv_f1(out_1))
        out_1 = torch.cat((out_0, out_1), dim=-3)

        out_2 = self.relu(self.conv_p2(out_1))
        out_2 = self.relu(self.conv_f2(out_2))
        out_2 = torch.cat((out_0, out_2), dim=-3)

        out_3 = self.relu(self.conv_p3(out_2))
        out_3 = self.relu(self.conv_f3(out_3))
        out_3 = torch.cat((out_0, out_3), dim=-3)

        out_4 = self.relu(self.conv_p4(out_3))
        out_4 = self.relu(self.conv_f4(out_4))
        out_4 = torch.cat((out_0, out_4), dim=-3)

        out_5 = self.relu(self.conv_p5(out_4))
        out_5 = self.relu(self.conv_f5(out_5))
        out_5 = torch.cat((out_0, out_5), dim=-3)

        out_6 = self.relu(self.conv_p6(out_5))
        out_6 = self.relu(self.conv_f6(out_6))
        out_6 = torch.cat((out_0, out_6), dim=-3)

        out_7 = self.relu(self.conv_p7(out_6))
        out_7 = self.relu(self.conv_f7(out_7))
        out_7 = torch.cat((out_0, out_7), dim=-3)

        out_8 = self.relu(self.conv_p8(out_7))
        out_8 = self.conv_f8(out_8)

        out = self.relu(x + out_8)  # x + out_8

        return out


class ResNet50FeatureExtractor(nn.Module):

    def __init__(self, blocks=[1, 2, 3, 4], pretrained=False, progress=True, **kwargs):
        super(ResNet50FeatureExtractor, self).__init__()
        self.model = models.resnet50(pretrained, progress, **kwargs)
        del self.model.avgpool
        del self.model.fc
        self.blocks = blocks

    def forward(self, x):
        feats = list()

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        if 1 in self.blocks:
            feats.append(x)

        x = self.model.layer2(x)
        if 2 in self.blocks:
            feats.append(x)

        x = self.model.layer3(x)
        if 3 in self.blocks:
            feats.append(x)

        x = self.model.layer4(x)
        if 4 in self.blocks:
            feats.append(x)

        return feats


class CompoundLoss(_Loss):

    def __init__(self, blocks=[1, 2, 3, 4], mse_weight=1, resnet_weight=0.01):
        super(CompoundLoss, self).__init__()

        self.mse_weight = mse_weight
        self.resnet_weight = resnet_weight

        self.blocks = blocks
        self.model = ResNet50FeatureExtractor(pretrained=True)

        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()

        self.criterion = nn.MSELoss()

    def forward(self, input, target):
        loss_value = 0

        input_feats = self.model(torch.cat([input, input, input], dim=1))
        target_feats = self.model(torch.cat([target, target, target], dim=1))

        feats_num = len(self.blocks)
        for idx in range(feats_num):
            loss_value += self.criterion(input_feats[idx], target_feats[idx])
        loss_value /= feats_num

        loss = self.mse_weight * self.criterion(input, target) + self.resnet_weight * loss_value

        return loss
