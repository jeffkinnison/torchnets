import time

import torch


class PerformanceMonitor(object):
    def __init__(self, metrics=None):
        self.metrics = metrics if metrics is not None else []
        self.metrics = list(set(self.metrics) + {'accuracy', 'loss', 'time'})
        self.train = {metric: [] for metric in metrics}
        self.validation = {metric: [] for metric in metrics}
        self.test = {metric: None for metric in metrics}

    def _init_accuracy(self):
        self.accuracy_correct = 0
        self.accuracy_total = 0

    def _init_loss(self):
        self.running_loss = 0
        self.n_batches = 0

    def _init_time(self):
        self.start_time = time.time()

    def _finalize_accuracy(self):
        return self.accuracy_correct / self.accuracy_total

    def _finalize_loss(self):
        return self.running_loss / self.n_batches

    def _finalize_time(self):
        return time.time() - self.start_time()

    def accuracy(self, **kwargs):
        preds, labels = kwargs['predictions'], kwargs['labels']
        self.accuracy_total += labels.size(0)
        _, predicted = torch.max(preds, 1)
        self.accuracy_correct += (predicted.data == labels.data).sum().item()

    def loss(self, **kwargs):
        loss_val = kwargs['loss']
        self.running_loss += loss_val
        self.n_batches += 1

    def batch_start(self):
        pass

    def batch_end(self, pred, label, loss):
        for m in self.metrics:
            metric_method = getattr(self, m)
            metric_method(pred=pred, label=label, loss=loss)

    def epoch_start(self):
        for m in self.metrics:
            getattr(self, '_init_' + m)()

    def epoch_end(self, mode):
        for m in self.metrics:
            val = getattr(self, '_finalize_' + m)()
            if mode == 'train':
                self.train[m].append(val)
            if mode == 'validation':
                self.validation[m].append(val)
            if mode == 'test':
                self.test[m] = val

    def to_json(self):
        return {
            'train': self.train,
            'validation': self.validation,
            'test': self.test
        }
