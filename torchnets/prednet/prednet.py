from torchnets.prednet.blocks import PredNetBlock

from torch.nn import Module, Conv2d
import torch.nn.functional as F


class PredNet(Module):
    def __init__(self):
        super(PredNet, self).__init__()
