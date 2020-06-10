import torch.nn as nn
from torch.nn.functional import relu as Relu


def Flatten(x):
    N, C, H, W = x.size() # read in N, C, H, W
    return x.view(N, -1)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        ksize = 3
        netPadding = (ksize - 1) // 2
        self.convInput = nn.Conv2d(1, 32, kernel_size=ksize, stride=1, padding=netPadding)
        self.conv32 = nn.Conv2d(32, 32, kernel_size=ksize, stride=1, padding=netPadding)
        self.conv32to64 = nn.Conv2d(32, 64, kernel_size=ksize, stride=1, padding=netPadding)
        self.conv64 = nn.Conv2d(64, 64, kernel_size=ksize, stride=1, padding=netPadding)
        self.conv64to128 = nn.Conv2d(64, 128, kernel_size=ksize, stride=1, padding=netPadding)
        self.conv128 = nn.Conv2d(128, 128, kernel_size=ksize, stride=1, padding=netPadding)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linearto128 = nn.Linear(2097152, 128)
        self.linearto1 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
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
        return x7
