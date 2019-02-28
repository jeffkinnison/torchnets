import torch
import torch.nn as nn
import torch.optim as optim

from torchnets.utils.result import PerformanceMonitor


class Model(object):
    def __init__(self, model, criterion, optimizer, **optimizer_kwargs):
        super(Model, self).__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer(model.parameters(), **optimizer_kwargs)
        self.performance = None

    def train(self, epochs, criterion, optimizer, training_set,
              validation_set=None, metrics=None, verbose=True):

        if self.performance is None:
            self.performance = PerformanceMonitor(metrics)

        for e in range(epochs):
            self.model.train()
            self.performance.epoch_start()

            for data in training_set:
                self.performance.batch_start()

                input, label = data
                if self.model.parameters()[0].is_cuda():
                    input = input.cuda()
                    label = label.cuda()

                optimizer.zero_grad()
                output = self.model(input)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()

                self.performance.batch_end(output, label, loss)

            self.performance.epoch_end('train')
            print("Epoch {}: {} training loss, {} training accuracy")

            if validation_set is not None:
                self.evaluate(validation_set, criterion, mode='validation',
                              verbose=verbose)

    def evaluate(self, dataset, criterion, mode='test', verbose=True):
        self.model.eval()
        with torch.no_grad():
            self.performance.epoch_start()
            for data in dataset:
                self.performance.batch_start()

                input, label = data
                if self.model.parameters()[0].is_cuda():
                    input = input.cuda()
                    label = label.cuda()

                output = self.model(input)
                loss = criterion(output, label)

                self.performance.batch_end(output, label, loss)
            self.epoch_end(mode)
