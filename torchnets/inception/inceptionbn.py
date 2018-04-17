from torchnets.utils import Conv2dBN

import torch
from torch.nn import Module, Linear
import torch.nn.functional as F


class InceptionBN(Module):
    """Implementation of the Inception net with batch normalization.

    An implementation of the Inception network (GoogLeNet), introduced in
    Szegedy *et al.*[1]_, with batch normalizaton as described in Ioffe and
    Szegedy [2]_.

    Parameters
    ----------
    in_channels : int
        The number of channels in the input data.
    n_classes: int
        The number of classes in the data.
    data_size : tuple of int
        The dimensions of the input images (height, width).

    Attributes
    ----------
    conv_1
    conv_2
    conv_3
    i3a
        Inception module 3a.
    i3b
        Inception module 3b.
    i4a
        Inception module 4a.
    i4b
        Inception module 4b.
    i4c
        Inception module 4c.
    i4d
        Inception module 4d.
    i4e
        Inception module 4e.
    i5a
        Inception module 5a.
    i5b
        Inception module 5b.
    fc

    References
    ----------
    .. [1] Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D.,
       Erhan, D., Vanhoucke, V., & Rabinovich, A. (2015, June). Going deeper
       with convolutions. CVPR.

    .. [2] Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating
       deep network training by reducing internal covariate shift. arXiv
       preprint arXiv:1502.03167.

    """

    def __init__(self, in_channels, n_classes, data_size):
        super(InceptionBN, self).__init__()

        self.conv_1 = Conv2dBN(in_channels, 64, 7, stride=2, padding=3)

        self.conv_2 = Conv2dBN(64, 192, 1)
        self.conv_3 = Conv2dBN(192, 192, 3, padding=1)

        self.i3a = InceptionModuleDR(192, 64, 96, 128, 16, 32, 32)
        self.i3b = InceptionModuleDR(256, 128, 128, 192, 32, 96, 64)

        self.i4a = InceptionModuleDR(480, 192, 96, 208, 16, 48, 64)
        self.i4b = InceptionModuleDR(512, 160, 112, 224, 24, 64, 64)
        self.i4c = InceptionModuleDR(512, 128, 128, 256, 24, 64, 64)
        self.i4d = InceptionModuleDR(512, 112, 144, 288, 32, 64, 64)
        self.i4e = InceptionModuleDR(528, 256, 160, 320, 32, 128, 128)

        self.i5a = InceptionModuleDR(832, 256, 160, 320, 32, 128, 128)
        self.i5b = InceptionModuleDR(832, 384, 192, 384, 48, 128, 128)

        self.fc1 = Linear(1024, 1000)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x, 3, stride=2)

        x = F.relu(self.conv_2(x))
        x = F.relu(self.conv_3(x))
        x = F.max_pool2d(x, 3, stride=2)

        x = self.i3a(x)
        x = self.i3b(x)
        x = F.max_pool2d(x, 3, stride=2)

        x = self.i4a(x)
        x = self.i4b(x)
        x = self.i4c(x)
        x = self.i4d(x)
        x = self.i4e(x)
        x = F.max_pool2d(x, 3, stride=2)

        x = self.i5a(x)
        x = self.i5b(x)
        x = F.max_pool2d(x, 3, stride=2)

        x = F.softmax(self.fc1(x))
        return x


class InceptionModule(Module):
    """Naive inception module.

    The first Inception module presented in Szegedy *et al.*[1]_.

    Parameters
    ----------
    in_channels : int
        The number of channels in the input data.
    channels_1x1 : int
        The number of channels output by the 1x1 convolutional layer.
    channels_3x3 : int
        The number of channels output by the 3x3 convolutional layer.
    channels_5x5 : int
        The number of channels output by the 5x5 convolutional layer.

    Attributes
    ----------
    conv_1x1
    conv_3x3
    conv_5x5

    References
    ----------
    .. [1] Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D.,
       Erhan, D., Vanhoucke, V., & Rabinovich, A. (2015, June). Going deeper
       with convolutions. CVPR.

    """

    def __init__(self, in_channels, channels_1x1, channels_3x3, channels_5x5):
        super(InceptionModule, self).__init__()

        self.conv_1x1 = Conv2dBN(in_channels, channels_1x1, 1)
        self.conv_3x3 = Conv2dBN(in_channels, channels_3x3, 3)
        self.conv_5x5 = Conv2dBN(in_channels, channels_5x5, 5)

    def forward(self, x):
        x1 = F.relu(self.conv_1x1(x))
        x2 = F.relu(self.conv_3x3(x))
        x3 = F.relu(self.conv_5x5(x))
        x4 = F.max_pool2d(x, 3, stride=1)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return x


class InceptionModuleDR(Module):
    """Inception module with dimensionality reduction.

    The second Inception module presented in Szegedy *et al.*[1]_.

    Parameters
    ----------
    in_channels : int
        The number of channels in the input data.
    channels_1x1 : int
        The number of channels output by the 1x1 convolutional layer.
    channels_3x3r : int
        The number of channels output by the 1x1 reduction convolutional
        layer on the 3x3 convolutional layer path.
    channels_3x3 : int
        The number of channels output by the 3x3 convolutional layer.
    channels_5x5r : int
        The number of channels output by the 1x1 reduction convolutional
        layer on the 5x5 convolutional layer path.
    channels_5x5 : int
        The number of channels output by the 5x5 convolutional layer.
    channels_poolr : int
        The number of channels output by the 1x1 reduction convolutional
        layer on the max pooling path.

    Attributes
    ----------
    conv_1x1
    conv_3x3r
    conv_3x3
    conv_5x5r
    conv_5x5
    poolr

    References
    ----------
    .. [1] Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D.,
       Erhan, D., Vanhoucke, V., & Rabinovich, A. (2015, June). Going deeper
       with convolutions. CVPR.

    """

    def __init__(self, in_channels, channels_1x1, channels_3x3r, channels_3x3,
                 channels_5x5r, channels_5x5, channels_poolr):
        super(InceptionModuleDR, self).__init__()

        self.conv_1x1 = Conv2dBN(in_channels, channels_1x1, 1)
        self.conv_3x3r = Conv2dBN(in_channels, channels_3x3r, 1)
        self.conv_3x3 = Conv2dBN(channels_3x3r, channels_3x3, 3, padding=1)
        self.conv_5x5r = Conv2dBN(in_channels, channels_5x5r, 1)
        self.conv_5x5 = Conv2dBN(channels_5x5r, channels_5x5, 5)
        self.poolr = Conv2dBN(in_channels, channels_poolr, 1)

    def forward(self, x):
        x1 = F.relu(self.conv_1x1(x))
        x2 = F.relu(self.conv_3x3(F.relu(self.conv_3x3r(x))))
        x3 = F.relu(self.conv_5x5(F.relu(self.conv_5x5r(x))))
        x4 = F.max_pool2d(self.poolr(x), 3, stride=1)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return x
