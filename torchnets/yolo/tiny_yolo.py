from torchnets.utils import Conv2dBN

from torch.nn import Module, Linear
import torch.nn.functional as F


class TinyYolo(Module):
    def __init__(self, in_channels, data_shape):
        super(TinyYolo, self).__init__()

        self.conv_1 = Conv2dBN(in_channels, 16, 3, padding=1)
        data_shape /= 2

        self.conv_2 = Conv2dBN(16, 32, 3, padding=1)
        data_shape /= 2

        self.conv_3 = Conv2dBN(32, 64, 3, padding=1)
        data_shape /= 2

        self.conv_4 = Conv2dBN(64, 128, 3, padding=1)
        data_shape /= 2

        self.conv_5 = Conv2dBN(128, 256, 3, padding=1)
        data_shape /= 2

        self.conv_6 = Conv2dBN(256, 512, 3, padding=1)
        data_shape /= 2

        self.conv_7 = Conv2dBN(512, 1024, 3, padding=1)
        self.conv_8 = Conv2dBN(1024, 256, 3, padding=1)
        self.dense_1 = Linear(data_shape.prod() * 256, 1470)

    def forward(self, x):
        x = F.leaky_relu(self.conv_1(x))
        x = F.max_pool2d(x, 2, stride=2)

        x = F.leaky_relu(self.conv_2(x))
        x = F.max_pool2d(x, 2, stride=2)

        x = F.leaky_relu(self.conv_3(x))
        x = F.max_pool2d(x, 2, stride=2)

        x = F.leaky_relu(self.conv_4(x))
        x = F.max_pool2d(x, 2, stride=2)

        x = F.leaky_relu(self.conv_5(x))
        x = F.max_pool2d(x, 2, stride=2)

        x = F.leaky_relu(self.conv_6(x))
        x = F.max_pool2d(x, 2, stride=2)

        x = F.leaky_relu(self.conv_7(x))
        x = F.leaky_relu(self.conv_8(x))

        x = self.dense_1(x.view(-1))
