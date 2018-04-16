from torch.nn import Module, Conv1d, Conv2d, Conv3d, BatchNorm1d, \
                     BatchNorm2d, BatchNorm3d


class ConvBN(Module):
    """Convolution followed by batch normalization.

    Convolutions with batch normalization as described by Ioffe and Szegedy[1]_

    Parameters
    ----------
    in_channels : int
        The number of channels in the input data.
    out_channels : int
        The number of channels to output from this layer.
    shape : int or tuple of int
        The shape of the convolutional filters.
    eps : float
        Epsilon for the denominator in batch normalization.

    Attributes
    ----------
    conv
        The convolutional layer.
    bn
        The batch normalization layer.

    References
    ----------
    .. [1] Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating
       deep network training by reducing internal covariate shift. arXiv
       preprint arXiv:1502.03167.

    """

    def forward(self, x):
        return self.bn(self.conv(x))


class Conv1dBN(ConvBN):
    def __init__(self, in_channels, out_channels, shape, eps=1e-5, **kws):
        super(Conv1dBN, self).__init__()
        self.conv = Conv1d(in_channels, out_channels, shape, **kws)
        self.bn = BatchNorm1d(out_channels, eps=eps)


class Conv2dBN(ConvBN):
    def __init__(self, in_channels, out_channels, shape, eps=1e-5, **kws):
        super(Conv2dBN, self).__init__()
        self.conv = Conv2d(in_channels, out_channels, shape, **kws)
        self.bn = BatchNorm2d(out_channels, eps=eps)



class Conv3dBN(ConvBN):
    def __init__(self, in_channels, out_channels, shape, eps=1e-5, **kws):
        super(Conv3dBN, self).__init__()
        self.conv = Conv3d(in_channels, out_channels, shape, **kws)
        self.bn = BatchNorm3d(out_channels, eps=eps)
