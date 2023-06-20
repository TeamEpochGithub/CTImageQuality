from torchvision.models import efficientnet_b0
import torch.nn as nn

from pretrain.pretrain_models.model_efficientnet_denoise import Decoder


class EfficientNetWithDecoder(nn.Module):
    def __init__(self):
        super(EfficientNetWithDecoder, self).__init__()

        self.efficientnet = efficientnet_b0()
        self.efficientnet = self.adapt_efficientnet_to_grayscale(self.efficientnet)
        self.efficientnet = nn.Sequential(*(list(self.efficientnet.children())[:-2]))

        self.decode4 = Decoder(1280, 256 + 256, 256)
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
        x = self.efficientnet(x)
        print(x.shape)
        x = self.decode4(x)
        x = self.decode3(x)
        x = self.decode2(x)
        x = self.decode1(x)
        x = self.decode0(x)
        x = self.conv_last(x)
        return x

    def adapt_efficientnet_to_grayscale(self, model):
        num_channels = 1  # Grayscale images have 1 channel
        out_channels = model.features[0][0].out_channels
        kernel_size = model.features[0][0].kernel_size
        stride = model.features[0][0].stride
        padding = model.features[0][0].padding
        model.features[0][0] = nn.Conv2d(num_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                         padding=padding, bias=False)
        return model
