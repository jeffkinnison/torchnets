import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG19(nn.Module):
    def __init__(self, in_channels, n_classes, data_shape):
        super(VGG19, self).__init__()

        if isinstance(data_shape, tuple):
            data_shape = torch.tensor(list(data_shape), dtype=torch.long)

        self.conv_1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv_2 = nn.Conv2d(64, 64, 3, padding=1)

        data_shape = data_shape / 2

        self.conv_3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv_4 = nn.Conv2d(128, 128, 3, padding=1)

        data_shape = data_shape / 2

        self.conv_5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv_6 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv_7 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv_8 = nn.Conv2d(256, 256, 3, padding=1)

        data_shape = data_shape / 2

        self.conv_9 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv_10 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv_11 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv_12 = nn.Conv2d(512, 512, 3, padding=1)

        data_shape = data_shape / 2

        self.conv_13 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv_14 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv_15 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv_16 = nn.Conv2d(512, 512, 3, padding=1)

        data_shape = data_shape / 2

        self.dense_1 = nn.Linear(512 * np.product(data_shape), 4096)
        self.dense_2 = nn.Linear(4096, 4096)
        self.dense_3 = nn.Linear(4096, n_classes)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv_3(x))
        x = F.relu(self.conv_4(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv_5(x))
        x = F.relu(self.conv_6(x))
        x = F.relu(self.conv_7(x))
        x = F.relu(self.conv_8(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv_9(x))
        x = F.relu(self.conv_10(x))
        x = F.relu(self.conv_11(x))
        x = F.relu(self.conv_12(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv_13(x))
        x = F.relu(self.conv_14(x))
        x = F.relu(self.conv_15(x))
        x = F.relu(self.conv_16(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.dense_1(x.flatten(start_dim=1)))
        x = F.relu(self.dense_2(x))
        x = F.softmax(self.dense_3(x), dim=1)

        return x


if __name__ == '__main__':
    import sys

    m = VGG19(1, 1000, (224, 224))
    x = torch.zeros(20, 1, 224, 224)

    if len(sys.argv) > 1 and sys.argv[1] == 'cuda':
        m = m.cuda()
        x = x.cuda()

    print('Model: {}'.format(m))
    print('Input Shape: {}'.format(x.size()))

    y = m(x)
    print('Output Shape: {}'.format(y.size()))
