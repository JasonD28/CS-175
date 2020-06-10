import torch
import torch.nn as nn
from torch.nn.functional import relu as Relu


def Flatten(x):
    N, C, H, W = x.size() # read in N, C, H, W
    return x.view(N, -1)


class Classifier_w_metadata(nn.Module):
    def __init__(self):
        super().__init__()
        ksize = 3
        netPadding = (ksize - 1) // 2
        self.convInput = nn.Conv2d(1, 32, kernel_size=ksize, stride=1, padding=netPadding)
        self.conv32 = nn.Conv2d(32, 32, kernel_size=ksize, stride=1, padding=netPadding)
        self.conv32to64 = nn.Conv2d(32, 64, kernel_size=ksize, stride=1, padding=netPadding)
        self.conv64 = nn.Conv2d(64, 64, kernel_size=ksize, stride=1, padding=netPadding)
        self.conv64to128 = nn.Conv2d(64, 128, kernel_size=ksize, stride=1, padding=netPadding)
        self.conv128 = nn.Conv2d(128, 128, kernel_size=ksize, stride=1, padding=netPadding)
        self.conv128to256 = nn.Conv2d(128, 256, kernel_size=ksize, stride=1, padding=netPadding)
        self.conv256 = nn.Conv2d(256, 256, kernel_size=ksize, stride=1, padding=netPadding)
        self.conv256to128 = nn.Conv2d(256, 128, kernel_size=ksize, stride=1, padding=netPadding)
        self.conv128to64 = nn.Conv2d(128, 64, kernel_size=ksize, stride=1, padding=netPadding)
        self.conv64to1 = nn.Conv2d(64, 1, kernel_size=ksize, stride=1, padding=netPadding)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pooling2 = nn.MaxPool2d(kernel_size=5, stride=2)
        self.upsampling = nn.Upsample(scale_factor=(2, 2))
        self.linearto128 = nn.Linear(2097152, 128)
        self.linearto2048 = nn.Linear(4096, 2048)
        self.linearto3 = nn.Linear(2048, 3)
        self.linearto1 = nn.Linear(128, 1)
        self.linear2to1 = nn.Linear(2, 1)
        self.linear1to1 = nn.Linear(1, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, data):
        # out = model(x) # Not sure what this does
        x = Relu(self.convInput(x), inplace=True)
        x = Relu(self.conv32(x), inplace=True)
        poolX = self.pooling(x)

        x2 = self.dropout(poolX)
        x2 = Relu(self.conv32to64(x2), inplace=True)
        x2 = Relu(self.conv64(x2), inplace=True)
        poolX2 = self.pooling(x2)

        x3 = self.dropout(poolX2)

        x4 = Relu(self.conv64to128(x3), inplace=True)
        x4 = Relu(self.conv128(x4), inplace=True)
        poolX4 = self.pooling(x4)
        # print(poolX4.shape)
        x5 = self.dropout(poolX4)
        # print(x5.shape)
        x5 = Flatten(x5)

        x6 = Relu(self.linearto128(x5), inplace=True)
        x6 = self.dropout(x6)

        # x7 = sigmoid(self.linearto1(x6))
        x7 = self.linearto1(x6)
        xD = data
        # print(x7.shape, xD.shape)
        x8 = torch.cat((x7, xD), dim=1)
        # print(x8)
        x9 = Relu(self.linear2to1(x8))
        x10 = self.linear1to1(x9)
        return x10
