import torch
import torch.nn as nn
from models.swin import StageModule
import torch.nn.functional as F

BN_MOMENTUM = True

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.LeakyReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)

        return out


class CLS_Block(nn.Module):
    def __init__(self, inplanes, planes):
        super(CLS_Block, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=2)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)

    def forward(self, x, x1):
        out = self.relu(self.bn1(self.conv1(x)))
        outs = self.bn2(self.conv2(out))
        outs += out
        return self.relu(outs)+x1


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_channels, cfg_swin, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()

        self.num_branches = num_branches
        self.cfg_swin = cfg_swin

        self.multi_scale_output = multi_scale_output
        self.num_inchannels = num_channels
        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.LeakyReLU(inplace=False)

    def downsample_branch(self, in_dim, out_dim):
        return StageModule(in_channels=in_dim, hidden_dimension=out_dim, layers=self.cfg_swin["layers"],
                                  downscaling_factor=2, num_heads=self.cfg_swin["num_heads"], head_dim=32,
                                  window_size=8, relative_pos_embedding=True)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels):
        layers = []
        for i in range(num_blocks[branch_index]):
            layers.append(block(num_channels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        if self.num_branches < 4:
            num_branches1 = num_branches+1
        else:
            num_branches1 = num_branches
        for i in range(num_branches1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        nn.BatchNorm2d(num_inchannels[i],
                                       momentum=True),
                        nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                self.downsample_branch(num_inchannels[j], num_outchannels_conv3x3)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3,
                                               momentum=True),
                                nn.LeakyReLU(inplace=False)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse

class HR_Transformer(nn.Module):

    def __init__(self, img_size=512, out_channels=1):
        super(HR_Transformer, self).__init__()

        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=True)
        self.relu = nn.ReLU(inplace=True)

        self.block = BasicBlock

        self.cfg_swin1 = {"layers": 2, "num_heads":4}
        self.stage1 = HighResolutionModule(1, self.block, [4], [64, 128], self.cfg_swin1)

        self.cfg_swin2 = {"layers": 2, "num_heads":8}
        self.stage2 = HighResolutionModule(2, self.block, [4, 4], [64, 128, 256], self.cfg_swin2)

        self.cfg_swin3 = {"layers": 2, "num_heads":16}
        self.stage3 = HighResolutionModule(3, self.block, [4, 4, 4], [64, 128, 256, 512], self.cfg_swin3)

        self.cfg_swin4 = {"layers": 2, "num_heads":32}
        self.stage4 = HighResolutionModule(4, self.block, [4, 4, 4, 4], [64, 128, 256, 512], self.cfg_swin4)

        self.cls_block1 = CLS_Block(64, 128)
        self.cls_block2 = CLS_Block(128, 256)
        self.cls_block3 = CLS_Block(256, 512)
        self.pool = nn.AvgPool2d(img_size//32)

        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, self.out_channels)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.stage1([x])
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        outs = self.cls_block1(x[0], x[1])
        outs = self.cls_block2(outs, x[2])
        outs = self.cls_block3(outs, x[3])
        outs = self.pool(outs).reshape(batch_size, -1)
        outs = self.relu(self.fc1(outs))
        if self.out_channels == 1:
            return 4 * F.sigmoid(self.fc2(outs))
        else:
            return self.fc2(outs)


