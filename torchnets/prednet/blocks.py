import torch.nn._functions.thnn.rnnFusedPointwise as fusedBackend
import torch.nn.functional as F

def ConvLSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    if input.is_cuda:
        igates = F.conv2d(input, w_ih)
        hgates = F.conv2d(hidden[0], w_hh)
        state = fusedBackend.LSTMFused.apply
        return state(igates, hgates, hidden[1]) if b_ih is None else state(igates, hgates, hidden[1], b_ih, b_hh)

    hx, cx = hidden
    gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)

    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = F.sigmoid(ingate)
    forgetgate = F.sigmoid(forgetgate)
    cellgate = F.tanh(cellgate)
    outgate = F.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * F.tanh(cy)

    return hy, cy
