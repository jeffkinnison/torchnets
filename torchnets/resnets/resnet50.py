import torch
from torch.nn import Module, Linear
import torch.nn.functional as F

from torchnets.resnets.blocks import ResBlock3
from torchnets.utils import Conv2dBN


class ResNet50(Module):
    def __init__(self, in_channels, n_classes):
        super(ResNet50, self).__init__()

        self.conv_1 = Conv2dBN(in_channels, 64, 7, stride=2, padding=3)

        self.block_21 = ResBlock3(64, 64, 256)
        self.block_22 = ResBlock3(256, 64, 256)
        self.block_23 = ResBlock3(256, 64, 256)

        self.block_31 = ResBlock3(256, 128, 512, downsample=True)
        self.block_32 = ResBlock3(512,  128, 512)
        self.block_33 = ResBlock3(512,  128, 512)
        self.block_34 = ResBlock3(512, 128, 512)

        self.block_41 = ResBlock3(512, 256, 1024, downsample=True)
        self.block_42 = ResBlock3(1024, 256, 1024)
        self.block_43 = ResBlock3(1024, 256, 1024)
        self.block_44 = ResBlock3(1024, 256, 1024)
        self.block_45 = ResBlock3(1024, 256, 1024)
        self.block_46 = ResBlock3(1024, 256, 1024)

        self.block_51 = ResBlock3(1024, 512, 2048, downsample=True)
        self.block_52 = ResBlock3(2048, 512, 2048)
        self.block_53 = ResBlock3(2048, 512, 2048)

        self.dense_1 = Linear(2048, n_classes)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x, 2, stride=2)
        x = self.block_21(x)
        x = self.block_22(x)
        x = self.block_23(x)
        x = self.block_31(x)
        x = self.block_32(x)
        x = self.block_33(x)
        x = self.block_34(x)
        x = self.block_41(x)
        x = self.block_42(x)
        x = self.block_43(x)
        x = self.block_44(x)
        x = self.block_45(x)
        x = self.block_46(x)
        x = self.block_51(x)
        x = self.block_52(x)
        x = self.block_53(x)
        x = F.avg_pool2d(x, 7, stride=1)
        x = x.flatten(start_dim=1)
        x = self.dense_1(x)
        return x


if __name__ == '__main__':
    import sys
    from torch.autograd import Variable

    m = ResNet50(1, 1000)
    x = torch.zeros(20, 1, 224, 224)

    if len(sys.argv) > 1 and sys.argv[1] == 'cuda':
        m = m.cuda()
        x = x.cuda()

    print('Model: {}'.format(m))
    print('Input Shape: {}'.format(x.size()))

    y = m(x)
    print('Output Shape: {}'.format(y.size()))
