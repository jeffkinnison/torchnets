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

    from torchnets.loaders.cifar10 import load_data

    m = LeNet5(1, 10, (32, 32))
    x = torch.zeros(20, 1, 32, 32)

    train_set, val_set, test_set = load_data('.', 128, True, None)

    if len(sys.argv) > 1 and sys.argv[1] == 'cuda':
        m = m.cuda()
        x = x.cuda()

    for e in range(100):
        current epoch = e
        running_loss = 0.0
        loss_denom = 0
        total = 0
        correct = 0
        for data in train_set:
            t_inputs, t_labels = data
            t_inputs, t_labels = Variable(t_inputs.cuda()), Variable(t_labels.cuda())

            loss_denom += 1
            total += t_labels.size(0)

            optimizer.zero_grad()
            t_outputs = model(t_inputs)
            loss = criterion(t_outputs, t_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
            _, predicted = torch.max(t_outputs, 1)
            correct += (predicted.data == t_labels.data).sum()
            print(optimizer.state_dict())
        train_loss.append(running_loss / loss_denom)
        train_acc.append(100 * correct / total)

        print('Epoch {}: train loss {}, train acc. {}'
              .format(current_epoch + 1, train_loss[-1], train_acc[-1]))

        if validation_set is not None:
            running_loss = 0.0
            loss_denom = 0
            total = 0
            correct = 0
            for data in validation_set:
                v_inputs, v_labels = data
                v_inputs = Variable(v_inputs.cuda(), volatile=True)
                v_labels = Variable(v_labels.cuda(), volatile=True)

                loss_denom += 1
                total += v_labels.size(0)

                v_outputs = model(v_inputs)
                loss = criterion(v_outputs, v_labels)
                running_loss += loss.data[0]
                _, predicted = torch.max(v_outputs.data, 1)
                correct += (predicted == v_labels.data).sum()

            val_loss.append(running_loss / loss_denom)
            val_acc.append(100 * correct / total)
            print('Epoch {}: val. loss {}, val. acc. {}'
                  .format(current_epoch + 1, val_loss[-1], val_acc[-1]))

    running_loss = 0
    loss_denom = 0
    total = 0
    correct = 0

    for data in test_set:
        inputs, labels = data
        inputs = Variable(inputs.cuda(), volatile=True)
        labels = Variable(labels.cuda(), volatile=True)

        loss_denom += 1
        total += labels.size(0)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels.data).sum()

    test_loss = running_loss / loss_denom
    test_acc = 100 * correct / total

    print('Test loss: {}\nTest acc.: {}'.format(test_loss, test_acc))
