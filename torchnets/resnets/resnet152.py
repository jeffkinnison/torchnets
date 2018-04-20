from torchnets.resnet.blocks import ResBlock3
from torchnets.utils import Conv2dBN

from torch.nn import Module, Linear
import torch.nn.functional as F


class ResNet101(Module):
    def __init__(self, in_channels, n_classes, data_shape):
        super(ResNet152, self).__init__()

        self.conv_1 = Conv2dBN(in_channels, 64, 7, stride=2, padding=1)

        data_shape /= 2

        self.block_21 = ResBlock3(in_channels, 64, 256)
        self.block_22 = ResBlock3(256, 64, 256)
        self.block_23 = ResBlock3(256, 64, 256)

        data_shape /= 2

        self.block_31 = ResBlock3(256, 128, 512, downsample=True)
        self.block_32 = ResBlock3(512,  128, 512)
        self.block_33 = ResBlock3(512,  128, 512)
        self.block_34 = ResBlock3(512, 128, 512)
        self.block_35 = ResBlock3(512,  128, 512)
        self.block_36 = ResBlock3(512,  128, 512)
        self.block_37 = ResBlock3(512, 128, 512)
        self.block_38 = ResBlock3(512, 128, 512)

        data_shape /= 2

        self.block_41 = ResBlock3(512, 256, 1024, downsample=True)
        self.block_42 = ResBlock3(1024, 256, 1024)
        self.block_43 = ResBlock3(1024, 256, 1024)
        self.block_44 = ResBlock3(1024, 256, 1024)
        self.block_45 = ResBlock3(1024, 256, 1024)
        self.block_46 = ResBlock3(1024, 256, 1024)
        self.block_47 = ResBlock3(1024, 256, 1024)
        self.block_48 = ResBlock3(1024, 256, 1024)
        self.block_49 = ResBlock3(1024, 256, 1024)
        self.block_410 = ResBlock3(1024, 256, 1024)
        self.block_411 = ResBlock3(1024, 256, 1024)
        self.block_412 = ResBlock3(1024, 256, 1024)
        self.block_413 = ResBlock3(1024, 256, 1024)
        self.block_414 = ResBlock3(1024, 256, 1024)
        self.block_415 = ResBlock3(1024, 256, 1024)
        self.block_416 = ResBlock3(1024, 256, 1024)
        self.block_417 = ResBlock3(1024, 256, 1024)
        self.block_418 = ResBlock3(1024, 256, 1024)
        self.block_419 = ResBlock3(1024, 256, 1024)
        self.block_420 = ResBlock3(1024, 256, 1024)
        self.block_421 = ResBlock3(1024, 256, 1024)
        self.block_422 = ResBlock3(1024, 256, 1024)
        self.block_423 = ResBlock3(1024, 256, 1024)
        self.block_424 = ResBlock3(1024, 256, 1024)
        self.block_425 = ResBlock3(1024, 256, 1024)
        self.block_426 = ResBlock3(1024, 256, 1024)
        self.block_427 = ResBlock3(1024, 256, 1024)
        self.block_428 = ResBlock3(1024, 256, 1024)
        self.block_429 = ResBlock3(1024, 256, 1024)
        self.block_430 = ResBlock3(1024, 256, 1024)
        self.block_431 = ResBlock3(1024, 256, 1024)
        self.block_432 = ResBlock3(1024, 256, 1024)
        self.block_433 = ResBlock3(1024, 256, 1024)
        self.block_434 = ResBlock3(1024, 256, 1024)
        self.block_435 = ResBlock3(1024, 256, 1024)
        self.block_436 = ResBlock3(1024, 256, 1024)

        data_shape /= 2

        self.block_51 = ResBlock3(1024, 512, 2048, downsample=True)
        self.block_52 = ResBlock3(2048, 512, 2048)
        self.block_53 = ResBlock3(2048, 512, 2048)

        data_shape /= 2

        self.dense_1 = Linear(2048 * data_shape.prod(), n_classes)

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
        x = self.block_47(x)
        x = self.block_48(x)
        x = self.block_49(x)
        x = self.block_410(x)
        x = self.block_411(x)
        x = self.block_412(x)
        x = self.block_413(x)
        x = self.block_414(x)
        x = self.block_415(x)
        x = self.block_416(x)
        x = self.block_417(x)
        x = self.block_418(x)
        x = self.block_419(x)
        x = self.block_420(x)
        x = self.block_421(x)
        x = self.block_422(x)
        x = self.block_423(x)
        x = self.block_424(x)
        x = self.block_425(x)
        x = self.block_426(x)
        x = self.block_427(x)
        x = self.block_428(x)
        x = self.block_429(x)
        x = self.block_430(x)
        x = self.block_431(x)
        x = self.block_432(x)
        x = self.block_433(x)
        x = self.block_434(x)
        x = self.block_435(x)
        x = self.block_436(x)
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

    m = ResNet152(1, 1000, (224, 224))
    x = torch.zeros(20, 1, 224, 224)

    if len(sys.argv) > 1 and sys.argv[1] == 'cuda':
        m = m.cuda()
        x = x.cuda()

    print('Model: {}'.format(m))
    print('Input Shape: {}'.format(x.size()))

    y = m(x)
    print('Output Shape: {}'.format(y.size()))
