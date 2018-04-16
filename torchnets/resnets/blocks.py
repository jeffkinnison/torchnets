from torchnets.utils import Conv2dBN

from torch.nn import Module
import torch.nn.functional as F


class ResBlock2(Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResBlock2, self).__init__()

        self.conv_1 = Conv2dBN(in_channels, out_channels, 3, padding=1)
        self.conv_2 = Conv2dBN(out_channels, out_channels, 3, padding=1)

        if downsample:
            self.downsample = Conv2dBN(in_channels, out_channels, 1, stride=2)
        else:
            self.downsample = None

    def forward(self, x):
        res = x
        x = F.relu(self.conv_1(x))
        x = self.conv_2(x)

        if self.downsample is not None:
            res = self.downsample(res)

        x = x + res
        return F.relu(x)


class ResBlock3(Module):
    def __init__(self, in_channels, mid_channels, out_channels,
                 downsample=False):
        super(ResBlock2, self).__init__()

        self.conv_1 = Conv2d(in_channels, mid_channels, 1)
        self.conv_2 = Conv2d(mid_channels, mid_channels, 3, padding=1)
        self.conv_3 = Conv2d(mid_channels, out_channels, 1)

        if downsample:
            self.downsample = Conv2dBN(in_channels, out_channels, 1, stride=2)
        else:
            self.downsample = None

    def forward(self, x):
        res = x
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = F.relu(self.conv_3(x))

        if self.downsample is not None:
            res = self.downsample(res)

        x = x + res
        return F.relu(x)
