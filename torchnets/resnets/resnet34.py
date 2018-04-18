from torchnets.resnet.blocks import ResBlock2
from torchnets.utils import Conv2dBN

from torch.nn import Module, Linear
import torch.nn.functional as F


class ResNet34(Module):
    def __init__(self, in_channels, n_classes, data_shape):
        super(ResNet34, self).__init__()

        self.conv_1 = Conv2dBN(in_channels, 64, 7, stride=2, padding=1)

        data_shape /= 2

        self.block_21 = ResBlock2(64, 64)
        self.block_22 = ResBlock2(64, 64)
        self.block_23 = ResBlock2(64, 64)

        self.block_31 = ResBlock2(64, 128, downsample=True)
        self.block_32 = ResBlock2(128, 128)
        self.block_33 = ResBlock2(128, 128)
        self.block_34 = ResBlock2(128, 128)

        data_shape /= 2

        self.block_41 = ResBlock2(128, 256, downsample=True)
        self.block_42 = ResBlock2(256, 256)
        self.block_43 = ResBlock2(256, 256)
        self.block_44 = ResBlock2(256, 256)
        self.block_45 = ResBlock2(256, 256)
        self.block_46 = ResBlock2(256, 256)

        data_shape /= 2

        self.block_51 = ResBlock2(256, 512, downsample=True)
        self.block_52 = ResBlock2(512, 512)
        self.block_53 = ResBlock2(512, 512)

        data_shape /= 2

        self.dense_1 = Linear(512 * data_shape.prod(), n_classes)

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
        x = x.view(-1, x.size()[1] * x.size()[2] * x.size()[3])
        x = self.dense_1(x)
        return x


if __name__ == '__main__':
    import sys
    from torch.autograd import Variable

    m = ResNet34(1, 1000, (224, 224))
    x = torch.zeros(20, 1, 224, 224)

    if len(sys.argv) > 1 and sys.argv[1] == 'cuda':
        m = m.cuda()
        x = x.cuda()

    print('Model: {}'.format(m))
    print('Input Shape: {}'.format(x.size()))

    y = m(x)
    print('Output Shape: {}'.format(y.size()))
