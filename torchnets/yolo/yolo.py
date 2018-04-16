from torchnets.utils import Conv2dBN, Conv2dLocal

from torch.nn import Module, Linear
import torch.nn.functional as F


class Yolo(Module):
    """
    """

    def __init__(self, in_channels, data_shape):
        super(Yolo, self).__init__()

        # Block 1
        conv_1 = Conv2dBN(in_channels, 64, 7, stride=2, padding=1)

        # Block 2
        conv_2 = Conv2dBN(64, 192, 3, padding=1)

        # Block 3
        conv_3 = Conv2dBN(192, 128, 1, padding=1)
        conv_4 = Conv2dBN(128, 256, 3, padding=1)
        conv_5 = Conv2dBN(256, 256, 1, padding=1)
        conv_6 = Conv2dBN(256, 512, 3, padding=1)

        # Block 4
        conv_7 = Conv2dBN(512, 256, 1, padding=1)
        conv_8 = Conv2dBN(256, 512, 3, padding=1)
        conv_9 = Conv2dBN(512, 256, 1, padding=1)
        conv_10 = Conv2dBN(256, 512, 3, padding=1)
        conv_11 = Conv2dBN(512, 256, 1, padding=1)
        conv_12 = Conv2dBN(256, 512, 3, padding=1)
        conv_13 = Conv2dBN(512, 256, 1, padding=1)
        conv_14 = Conv2dBN(256, 512, 3, padding=1)
        conv_15 = Conv2dBN(512, 512, 1, padding=1)
        conv_16 = Conv2dBN(1024, 512, 3, padding=1)

        # Block 5
        conv_17 = Conv2dBN(512, 1024, 3, padding=1)
        conv_18 = Conv2dBN(1024, 512, 3, padding=1)
        conv_19 = Conv2dBN(512, 1024, 3, padding=1)
        conv_20 = Conv2dBN(1024, 512, 3, padding=1)

        # Classification block
        dense_classify = Linear(512, n_classes)

        # Detection block
        conv_21 = Conv2dBN(1024, 1024, 3, padding=1)
        conv_22 = Conv2dBN(1024, 1024, 3, stride=2, padding=1)
        conv_23 = Conv2dBN(1024, 1024, 3, padding=1)
        conv_24 = Conv2dBN(1024, 1024, 3, padding=1)
        local_1 = Conv2dLocal(7, 7, 1024, 256, 3, padding=1)
        dense_detect = Linear


    def forward(self, x):
        x = F.leaky_relu(self.conv_1(x))
        x = F.max_pool2d(x, 2, stride=2)

        x = F.leaky_relu(self.conv_2(x))
        x = F.max_pool2d(x, 2, stride=2)

        x = F.leaky_relu(self.conv_3(x))
        x = F.leaky_relu(self.conv_4(x))
        x = F.leaky_relu(self.conv_5(x))
        x = F.leaky_relu(self.conv_6(x))
        x = F.max_pool2d(x, 2, stride=2)

        x = F.leaky_relu(self.conv_7(x))
        x = F.leaky_relu(self.conv_8(x))
        x = F.leaky_relu(self.conv_9(x))
        x = F.leaky_relu(self.conv_10(x))
        x = F.leaky_relu(self.conv_11(x))
        x = F.leaky_relu(self.conv_12(x))
        x = F.leaky_relu(self.conv_13(x))
        x = F.leaky_relu(self.conv_14(x))
        x = F.leaky_relu(self.conv_15(x))
        x = F.leaky_relu(self.conv_16(x))
        x = F.max_pool2d(x, 2, stride=2)
