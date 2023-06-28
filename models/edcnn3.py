import torch
import torch.nn as nn
import torch.nn.functional as F

from models.edcnn import SobelConv2d


class EDCNN3(nn.Module):

    def __init__(self, in_ch=1, out_ch=32, sobel_ch=32):
        super(EDCNN3, self).__init__()

        self.conv_sobel = SobelConv2d(in_ch, sobel_ch, kernel_size=3, stride=1, padding=1, bias=True)

        self.first_sequence = nn.Sequential(
            # nn.BatchNorm2d(in_ch + sobel_ch),
            nn.Conv2d(in_ch + sobel_ch, out_ch, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU()
        )

        self.sequences = nn.ModuleList([
            nn.Sequential(
                # nn.BatchNorm2d(in_ch + sobel_ch + out_ch),
                nn.Conv2d(in_ch + sobel_ch + out_ch, out_ch, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU()
            ) for _ in range(5)  # Number of sequences
        ])

        self.final_sequence = nn.Sequential(
            # nn.BatchNorm2d(in_ch + sobel_ch + out_ch),
            nn.Conv2d(in_ch + sobel_ch + out_ch, out_ch, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(out_ch, in_ch, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU()
        )

        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.3)

        self.avgpool = nn.AdaptiveAvgPool2d((16, 16))
        self.fc1 = nn.Linear(16*16, 1)

        # self.fc2 = nn.Linear(nodes, 1)  # Final output node

    def forward(self, x):
        sobel_out = self.conv_sobel(x)
        out = torch.cat((x, sobel_out), dim=1)  # Concatenate along the channel dimension

        out = self.first_sequence(out)
        out = torch.cat((out, x, sobel_out), dim=1)

        for seq in self.sequences:
            out = seq(out)
            out = self.dropout(out)
            out = torch.cat((out, x, sobel_out), dim=1)

        out = self.final_sequence(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)  # Flatten the output
        out = self.dropout(out)
        out = self.fc1(out)
        out = torch.sigmoid(out) * 4

        return out