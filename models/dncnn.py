import torch
import torch.nn as nn


class DnCNN(nn.Module):
    def __init__(self, channels=1, num_of_layers=8):  # 17
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = [nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                            bias=False),
                  nn.ReLU(inplace=True)]
        for _ in range(num_of_layers - 2):
            layers.append(
                nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding,
                                bias=False))
        self.dncnn = nn.Sequential(*layers)

        self.relu = nn.LeakyReLU()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(channels, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out = self.dncnn(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)  # Flatten the output
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        out = torch.sigmoid(out) * 4

        return out
