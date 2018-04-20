import torch


class Whiten(object):
    """Perform ZCA whitening on an image.

    Parameters
    ----------
    epsilon : float
        Small adjust to prevent divide-by-zero.

    Attributes
    ----------
    epsilon : float
        Small adjust to prevent divide-by-zero.
    """
    def __init__(epsilon):
        self.epsilon = epsilon

    def __call__(self, tensor):
        return whiten(tensor, self.epsilon)


def whiten(x, e):
    """Perform ZCA whitening on an image.

    Parameters
    ----------
    x : `numpy.ndarray` or `PIL
    epsilon : float
        Small adjust to prevent divide-by-zero.

    Attributes
    ----------
    epsilon : float
        Small adjust to prevent divide-by-zero.
    """
    x = x - x.mean(0).repeat(x.size()[1], 1)
    sigma = x * x.t / x.shape[1]
    u, s, _ = torch.svd(sigma, some=False)
    return u * torch.diag(1 / torch.sqrt(S.diag() + e)) * u.t * x
