import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3d(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, kernel_size=None,
                 stride=1, padding=0, dilation=1, groups=1, activation=None,
                 dropout=None, epsilon=None, momentum=None):
        self.layer = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=True)
        self.activation = getattr(F, activation) if activation is not None \
                          else None
        self.dropout = dropout
        if epsilon is not None or momentum is not None:
            self.bn = nn.BatchNorm3d(out_channels, epsilon, momentum)
        else:
            self.bn = None

    def forward(self, x):
        x = self.layer(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.dropout is not None:
            x = F.dropout(x, self.dropout)
        if self.bn is not None:
            x = self.bn(x)
        return x
