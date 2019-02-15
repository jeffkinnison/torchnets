from torch.nn import Module
import torch.nn.functional as F

from torchnets.utils import Conv2dBN


class ResBlock2(Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResBlock2, self).__init__()

        if downsample:
            self.downsample = Conv2dBN(int(in_channels / 2), out_channels, 1, stride=2)
        else:
            self.downsample = None


        self.conv_1 = Conv2dBN(in_channels, out_channels, 3, padding=1)
        self.conv_2 = Conv2dBN(out_channels, out_channels, 3, padding=1)

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        res = x

        x = F.relu(self.conv_1(x))
        x = self.conv_2(x)

        x = x + res
        return F.relu(x)


class ResBlock3(Module):
    def __init__(self, in_channels, mid_channels, out_channels,
                 downsample=False):
        super(ResBlock3, self).__init__()

        if downsample:
            self.downsample = Conv2dBN(in_channels, mid_channels, 1, stride=2)
            self.conv_1 = Conv2dBN(mid_channels, mid_channels, 1)
            self.res_conv = Conv2dBN(mid_channels, out_channels, 1)
        else:
            self.downsample = None
            self.conv_1 = Conv2dBN(in_channels, mid_channels, 1)
            self.res_conv = Conv2dBN(in_channels, out_channels, 1)

        self.conv_2 = Conv2dBN(mid_channels, mid_channels, 3, padding=1)
        self.conv_3 = Conv2dBN(mid_channels, out_channels, 1)

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        res = self.res_conv(x)

        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = F.relu(self.conv_3(x))

        x = x + res
        return F.relu(x)
