import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(Module):
    def __init__(self, in_channels, n_classes, data_shape):
        super(LeNet5, self).__init__()

        if isinstance(data_shape, tuple):
            data_shape = torch.tensor(list(data_shape), dtype=torch.long)

        self.conv_1 = nn.Conv2d(in_channels, 6, 5, padding=2)
        data_shape /= 2

        self.conv_2 = nn.Conv2d(6, 16, 5, padding=2)
        data_shape /= 2

        self.dense_1 = nn.Linear(data_shape.prod() * 16, 120)
        self.dense_2 = nn.Linear(120, 84)
        self.dense_3 = nn.Linear(84, n_classes)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv_2(x))
        x = F.max_pool2d(x, 2)

        x = x.flatten(start_dim=1)
        x = F.relu(self.dense_1(x))
        x = F.relu(self.dense_2(x))
        x = self.dense_3(x)

        return x


if __name__ == '__main__':
    import sys

    model = LeNet5(3, 10, (32, 32))
    x = torch.zeros(20, 1, 32, 32)

    if len(sys.argv) > 1 and sys.argv[1] == 'cuda':
        model = model.cuda()
        x = x.cuda()

    print('Model: {}'.format(m))
    print('Input Shape: {}'.format(x.size()))
    y = m(x)
    print('Output Shape: {}'.format(y.size()))
