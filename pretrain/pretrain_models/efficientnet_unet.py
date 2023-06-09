import torch
import torch.nn as nn
import torchvision.models as models

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class EfficientNet_Denoise(nn.Module):
    def __init__(self, n_class=1, pretrained=True):
        super().__init__()
        self.encoder = models.efficientnet_b0(pretrained=pretrained)
        # # self.base_layers = list(self.encoder.children())
        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.SiLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            # DoubleConv(1280, 320),
            nn.ConvTranspose2d(1280, 320, kernel_size=2, stride=2),
            DoubleConv(320 + 112, 160),
            nn.ConvTranspose2d(160, 80, kernel_size=2, stride=2),
            DoubleConv(80 + 40, 80),
            nn.ConvTranspose2d(80, 40, kernel_size=2, stride=2),
            DoubleConv(40 + 24, 40),
            nn.ConvTranspose2d(40, 20, kernel_size=2, stride=2),
            DoubleConv(20 + 16, 20),
            nn.ConvTranspose2d(20, 10, kernel_size=2, stride=2)
        )

        self.conv_last = nn.Conv2d(10, n_class, 1)

    def forward(self, x):
        # Encoder
        x = self.layer0(x)
        features = []
        for name, module in self.encoder.named_children():
            if name == 'features':
                for block in module[1:]:
                    x = block(x)
                    features.append(x)

        skip_connections = [features[4], features[2], features[1], features[0]]  # Select desired skip connections

        # Decoder
        x = self.decoder[0](features[-1])
        for i, module in enumerate(self.decoder[1:], start=1):
            if i % 2 == 1:
                x = torch.cat([x, skip_connections[i // 2]], dim=1)
            x = module(x)
        x = self.conv_last(x)
        return x

# ins = torch.randn(8, 1, 512, 512)
# model = EfficientNet_Denoise()
# print(model(ins).shape)
