from torch.nn import Module, Conv2d, Linear, MaxPooling2d
import torch.nn.functional as F


class LeNet5(Module):
    def __init__(self, in_channels, n_classes, data_shape):
        self.conv_1 = Conv2d(in_channels, 6, 5)
        data_shape -= 4

        self.conv_2 = Conv2d(6, 16, 5)
        data_shape = (data_shape - 4) / 2

        self.dense_1 = Linear(data_shape.prod() * 16, 120)
        self.dense_2 = Linear(120, 84)
        self.dense_3 = Linear(84, n_classes)

    def forward(self, x):
        x = F.relu(self.conv_1)
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv_2(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.dense_1(x.view(-1)))
        x = F.relu(self.dense_2(x))
        x = self.dense_3(x)

        return x
