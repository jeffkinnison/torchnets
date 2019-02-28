import torch
import torch.nn as nn
import torch.nn.functional as F

import torchnets.layers as layers


class LeNet5(nn.Module):
    def __init__(self, in_channels, n_classes, data_shape,
                 hyperparameters=None):
        super(LeNet5, self).__init__()

        hyperparameters = hyperparameters if hyperparameters is not None \
                          else {}
        defaults = self.default_hyperparameters()
        defaults.update(hyperparameters)
        hyperparameters = defaults

        if isinstance(data_shape, tuple):
            data_shape = torch.tensor(list(data_shape), dtype=torch.long)

        self.conv_1 = layers.Conv2d(in_channels=in_channels,
                                    **hyperparameters['conv_1'])
        data_shape /= 2

        self.conv_2 = layers.Conv2d(**hyperparameters['conv_2'])
        data_shape /= 2

        self.dense_1 = layers.Linear(in_features=data_shape.prod() * 16,
                                     **hyperparameters['linear_1'])
        self.dense_2 = layers.Linear(**hyperparameters['linear_2'])
        self.dense_3 = nn.Linear(out_features=n_classes,
                                 **hyperparameters['linear_3'])


    def default_hyperparameters(self):
        return {
            'conv_1': {
                'out_channels': 6,
                'kernel_size': 5,
                'padding': 2,
                'activation': 'relu',
            },
            'conv_2': {
                'in_channels': 6,
                'out_channels': 16,
                'kernel_size': 5,
                'padding': 2,
                'activation': 'relu',
            },
            'linear_1': {
                'out_features': 120,
                'activation': 'relu',
            },
            'linear_2': {
                'in_features': 120,
                'out_features': 84,
                'activation': 'relu',
            },
            'linear_3': {
                'in_features': 84
            }
        }


    def forward(self, x):
        x = self.conv_1(x)
        x = F.max_pool2d(x, 2)

        x = self.conv_2(x)
        x = F.max_pool2d(x, 2)

        x = x.flatten(start_dim=1)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)

        return x


if __name__ == '__main__':
    import sys

    model = LeNet5(3, 10, (32, 32))
    x = torch.zeros(20, 1, 32, 32)

    if len(sys.argv) > 1 and sys.argv[1] == 'cuda':
        model = model.cuda()
        x = x.cuda()

    print('Model: {}'.format(model))
    print('Input Shape: {}'.format(x.size()))
    y = model(x)
    print('Output Shape: {}'.format(y.size()))
