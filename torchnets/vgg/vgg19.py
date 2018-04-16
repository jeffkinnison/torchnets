from torch.nn import Module, Conv2d, Linear, MaxPooling2d
import torch.nn.functional as F


class VGG19(Module):
    def __init__(self, in_channels, n_classes, data_shape):
        super(VGG19, self).__init__()

        self.conv_1 = Conv2d(in_channels, 64, 3, padding=1)
        self.conv_2 = Conv2d(64, 64, 3, padding=1)

        data_shape = data_shape / 2

        self.conv_3 = Conv2d(64, 128, 3, padding=1)
        self.conv_4 = Conv2d(128, 128, 3, padding=1)

        data_shape = data_shape / 2

        self.conv_5 = Conv2d(128, 256, 3, padding=1)
        self.conv_6 = Conv2d(256, 256, 3, padding=1)
        self.conv_7 = Conv2d(256, 256, 3, padding=1)
        self.conv_8 = Conv2d(256, 256, 3, padding=1)

        data_shape = data_shape / 2

        self.conv_9 = Conv2d(256, 512, 3, padding=1)
        self.conv_10 = Conv2d(512, 512, 3, padding=1)
        self.conv_11 = Conv2d(512, 512, 3, padding=1)
        self.conv_12 = Conv2d(512, 512, 3, padding=1)

        data_shape = data_shape / 2

        self.conv_13 = Conv2d(512, 512, 3, padding=1)
        self.conv_14 = Conv2d(512, 512, 3, padding=1)
        self.conv_15 = Conv2d(512, 512, 3, padding=1)
        self.conv_16 = Conv2d(512, 512, 3, padding=1)

        data_shape = data_shape / 2

        self.dense_1 = Linear(np.product(data_shape), 4096)
        self.dense_2 = Linear(4096, 4096)
        self.dense_3 = Linear(4096, n_classes)

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

        x = F.relu(self.dense_1(x.view(-1)))
        x = F.relu(self.dense_2(x))
        x = F.softmax(self.dense_3(x))

        return x
