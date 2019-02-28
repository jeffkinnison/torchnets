import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self, in_features=None, out_features=None, activation=None,
                 dropout=None):
        self.layer = nn.Linear(in_features, out_features, bias=True)
        self.activation = getattr(F, activation) if activation is not None \
                          else None
        self.dropout = dropout

    def forward(self, x):
        x = self.layer(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.dropout is not None:
            x = F.dropout(x, self.dropout)
        return x
