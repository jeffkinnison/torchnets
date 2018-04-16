from math import floor

import torch
from torch.nn import Module
import torch.nn.functional as F


class ConvLSTMCell(Module):
    def __init__(self, input_shape, input_channels, hidden_channels,
                 kernel_size, bias=True):

        super(ConvLSTMCell, self).__init__()

        self.conv_1 = Conv2d(input_channels + hidden_channels,
                             4 * hidden_channels,
                             kernel_size,
                             padding=int(floor(kernel_size / 2)),
                             bias=bias)

    def forward(self, x, state):
        h, c = state

        cat = torch.cat([x, h], dim=1)

        cat_conv = self.conv_1(cat)
        i, f, o, g = torch.split(cat_conv, int(cat_conv.size()[0] / 4), dim=1)
        i = F.sigmoid(i)
        f = F.sigmoid(f)
        o = F.sigmoid(o)
        g = F.tanh(g)

        c = f * c + i * g
        h = o * F.tanh(c)

        return h, c


class ConvLSTM(Module):
    def __init__(self):
        super(ConvLSTM, self).__init__()
