import torch
from torch.nn import Module, Conv2d, Linear
import torch.nn.functional as F


class LeNet5(Module):
    def __init__(self, in_channels, n_classes, data_shape):
        super(LeNet5, self).__init__()

        if isinstance(data_shape, tuple):
            data_shape = torch.LongTensor([i for i in data_shape])

        self.conv_1 = Conv2d(in_channels, 6, 5, padding=2)
        data_shape /= 2

        self.conv_2 = Conv2d(6, 16, 5, padding=2)
        data_shape /= 2

        print(data_shape.prod() * 16)
        self.dense_1 = Linear(data_shape.prod() * 16, 120)
        self.dense_2 = Linear(120, 84)
        self.dense_3 = Linear(84, n_classes)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv_2(x))
        x = F.max_pool2d(x, 2)

        x = x.view(-1, x.size()[1] * x.size()[2] * x.size()[3])
        x = F.relu(self.dense_1(x))
        x = F.relu(self.dense_2(x))
        x = self.dense_3(x)

        return x


if __name__ == '__main__':
    import sys
    from torch.autograd import Variable

    m = LeNet5(1, 10, (32, 32))
    x = torch.zeros(20, 1, 32, 32)

    if len(sys.argv) > 1 and sys.argv[1] == 'cuda':
        m = m.cuda()
        x = x.cuda()

    print('Model: {}'.format(m))
    print('Input Shape: {}'.format(x.size()))
    y = m(Variable(x))
    print('Output Shape: {}'.format(y.size()))
