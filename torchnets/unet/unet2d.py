import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    """Create a U-Net model using 2D operations.

    This model is based on the U-Net model described by Ronneberger *et al*
    [UNET]_ for solving membrane segmentation in electron microscopic images.

    Parameters
    ----------
    in_channels : int
        The number of channels in the input data (e.g., 1 for grayscale, 3 for
        color images).
    min_filters : int
        The number of filters to use at the topmost layers of U-Net. This is
        automatically doubled at lower layer block to preserve the original
        U-Net structure.

    Attributes
    ----------
    input : `ConvBlock`
        The input set of convolutional layers.
    down1 : `DownBlock`
        The first downsampling level.
    down2 : `DownBlock`
        The second downsampling level.
    down3 : `DownBlock`
        The third downsampling level.
    down4 : `DownBlock`
        The fourth downsampling level.
    up1 : `UpBlock`
        The first upsampling level.
    up2 : `UpBlock`
        The second upsampling level.
    up3 : `UpBlock`
        The third upsampling level.
    up4 : `UpBlock`
        The fourth upsampling level.
    out : `OutBlock`
        The output layer.

    References
    ----------
    .. [UNET] Ronneberger, O., Fischer, P., & Brox, T. (2015, October). U-net:
       Convolutional networks for biomedical image segmentation. In
       International Conference on Medical image computing and computer-
       assisted intervention (pp. 234-241). Springer, Cham.
    """

    def __init__(self, in_channels, min_filters=64):
        super(UNet, self).__init__()

        prev_filters = in_channels
        curr_filters = min_filters
        self.input = ConvBlock(prev_filters, curr_filters, 3)

        prev_filters = curr_filters
        curr_filters = int(curr_filters * 2)
        self.down1 = DownBlock(prev_filters, curr_filters, 3)

        prev_filters = curr_filters
        curr_filters = int(curr_filters * 2)
        self.down2 = DownBlock(prev_filters, curr_filters, 3)

        prev_filters = curr_filters
        curr_filters = int(curr_filters * 2)
        self.down3 = DownBlock(prev_filters, curr_filters, 3)

        prev_filters = curr_filters
        curr_filters = int(curr_filters * 2)
        self.down4 = DownBlock(prev_filters, curr_filters, 3)

        prev_filters = curr_filters
        curr_filters = int(curr_filters / 2)
        self.up1 = UpBlock(prev_filters, curr_filters, 3)

        prev_filters = curr_filters
        curr_filters = int(curr_filters / 2)
        self.up2 = UpBlock(prev_filters, curr_filters, 3)

        prev_filters = curr_filters
        curr_filters = int(curr_filters / 2)
        self.up3 = UpBlock(prev_filters, curr_filters, 3)

        prev_filters = curr_filters
        curr_filters = int(curr_filters / 2)
        self.up4 = UpBlock(prev_filters, curr_filters, 3)

        self.out = OutBlock(curr_filters)

    def forward(self, x):
        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)

        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)

        return x


class ConvBlock(nn.Module):
    """Convolutional block used at each depth layer in U-Net.

    Consists of two back-to-back convolutional layers with the same
    hyperparameters.

    Parameters
    ----------
    in_channels : int
        The number of input channels in the data.
    out_channels : int
        The number of channels for these layers to output.
    shape : int or tuple of int
        The shape of the convolutional filters.

    Attributes
    ----------
    conv1 : `torch.nn.Conv2d`
        The first convolutional layer in the block.
    conv2 : `torch.nn.Conv2d`
        The second convolutional layer in the block.
    """
    def __init__(self, in_channels, out_channels, shape):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, shape, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, shape, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class DownBlock(nn.Module):
    """Downsampling block used to create a new depth layer in U-Net.

    Consists of a 2x2 max pooling operation then two back-to-back convolutional
    layers with the same hyperparameters.

    Parameters
    ----------
    in_channels : int
        The number of input channels in the data.
    out_channels : int
        The number of channels for these layers to output.
    shape : int or tuple of int
        The shape of the convolutional filters.

    Attributes
    ----------
    conv : `ConvBlock`
        The first convolutional layer in the block.
    """
    def __init__(self, in_channels, out_channels, shape):
        super(DownBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels, shape)

    def forward(self, x):
        x = F.max_pool2d(x, (2, 2))
        x = self.conv(x)
        return x


class UpConv(nn.Module):
    """Up-Conv block to return from a depth layer in U-Net.

    Consists of a 2x2 upsampling operation then a 2x2 convolutional layer that
    maintains the same number of convolutional filters.

    Parameters
    ----------
    in_channels : int
        The number of input channels in the data.
    out_channels : int
        The number of channels for these layers to output.
    shape : int or tuple of int
        The shape of the convolutional filters.

    Attributes
    ----------
    upsample : `torch.nn.Upsample`
        Upsample the data in 2d.
    pad : `torch.nn.ReplicationPad2d`
        Pad the data to maintain shape. a 2x2 convolution shaves one pixel off
        of the input data in each dimension. `torch.nn.Conv2d` layers do not
        currently support single-pixel padding, so this extra layer is needed.
    conv : `torch.nn.Conv2d`
        The convolutional layer that operates on the upsampled data.
    """
    def __init__(self, in_channels, out_channels, shape):
        super(UpConv, self).__init__()
        self.upsample = nn.Upsample(scale_factor=(2, 2), mode='bilinear')
        self.pad = nn.ReplicationPad2d((0, 1, 0, 1))
        self.conv = nn.Conv2d(in_channels, out_channels, shape)

    def forward(self, x):
        x = self.upsample(x)
        x = self.pad(x)
        x = F.relu(self.conv(x))
        return x


class UpBlock(nn.Module):
    """Upsampling block used to return from a depth layer in U-Net.

    Consists of a 2x2 upconv operation then two back-to-back convolutional
    layers with the same hyperparameters.

    Parameters
    ----------
    in_channels : int
        The number of input channels in the data.
    out_channels : int
        The number of channels for these layers to output.
    shape : int or tuple of int
        The shape of the convolutional filters.

    Attributes
    ----------
    upconv : `UpConv`
        The upsampling operation.
    conv : `ConvBlock`
        The convolutional block.
    """
    def __init__(self, in_channels, out_channels, shape):
        super(UpBlock, self).__init__()
        self.upconv = UpConv(in_channels, in_channels, 2)
        self.conv = ConvBlock(in_channels + out_channels, out_channels, shape)

    def forward(self, x, x2):
        x = self.upconv(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv(x)
        return x


class OutBlock(nn.Module):
    """Output block in U-Net.

    Consists of a 1x1 convolution with a single filter, maps the data back to
    a single image.

    Parameters
    ----------
    in_channels : int
        The number of input channels in the data.

    Attributes
    ----------
    conv : `torch.nn.Conv2d`
        The out convolution.
    """
    def __init__(self, in_channels):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


if __name__ == '__main__':
    m = UNet(1)
