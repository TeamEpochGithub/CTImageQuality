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
    def __init__(self, mode, n_class=1, pretrained=True):
        super().__init__()
        self.encoders = {
            "b0": models.efficientnet_b0(pretrained=pretrained),
            "b1": models.efficientnet_b1(pretrained=pretrained),
            "b2": models.efficientnet_b2(pretrained=pretrained),
            "b3": models.efficientnet_b3(pretrained=pretrained),
            "b4": models.efficientnet_b4(pretrained=pretrained),
            "b5": models.efficientnet_b5(pretrained=pretrained),
            "b6": models.efficientnet_b6(pretrained=pretrained),
            "b7": models.efficientnet_b7(pretrained=pretrained),
        }

        self.model_features = {
            "b0": [16, 24, 40, 80, 112, 192, 320, 1280],
            "b1": [16, 24, 40, 80, 112, 192, 320, 1280],
            "b2": [16, 24, 48, 88, 120, 208, 352, 1408],
            "b3": [24, 32, 48, 96, 136, 232, 384, 1536],
            "b4": [24, 32, 56, 112, 160, 272, 448, 1792],
            "b5": [24, 40, 64, 128, 176, 304, 512, 2048],
            "b6": [32, 40, 72, 144, 200, 344, 576, 2304],
            "b7": [32, 48, 80, 160, 224, 384, 640, 2560],
        }

        self.layer0_features = {
            "b0": 32,
            "b1": 32,
            "b2": 32,
            "b3": 40,
            "b4": 48,
            "b5": 48,
            "b6": 56,
            "b7": 64
        }

        self.encoder = self.encoders[mode]
        self.model_feature = self.model_features[mode]

        self.layer0 = nn.Sequential(
            nn.Conv2d(1, self.layer0_features[mode], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(self.layer0_features[mode], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.SiLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.model_feature[7], self.model_feature[6], kernel_size=2, stride=2),
            DoubleConv(self.model_feature[6] + self.model_feature[4], self.model_feature[6]//2),
            nn.ConvTranspose2d(self.model_feature[6]//2, self.model_feature[3], kernel_size=2, stride=2),
            DoubleConv(self.model_feature[3] + self.model_feature[2], self.model_feature[3]),
            nn.ConvTranspose2d(self.model_feature[3], self.model_feature[2], kernel_size=2, stride=2),
            DoubleConv(self.model_feature[2] + self.model_feature[1], self.model_feature[2]),
            nn.ConvTranspose2d(self.model_feature[2], self.model_feature[1], kernel_size=2, stride=2),
            DoubleConv(self.model_feature[1] + self.model_feature[0], self.model_feature[1]),
            nn.ConvTranspose2d(self.model_feature[1], self.model_feature[0], kernel_size=2, stride=2)
        )

        self.conv_last = nn.Conv2d(self.model_feature[0], n_class, 1)

    def forward(self, x):
        # Encoder
        x = self.layer0(x)
        features = []
        for name, module in self.encoder.named_children():
            if name == 'features':
                for block in module[1:]:
                    x = block(x)
                    features.append(x)
        # for i in range(len(features)):
        #     print(i, features[i].shape)

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
# model = EfficientNet_Denoise(mode="b1")
# print(model(ins).shape)
