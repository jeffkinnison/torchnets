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
    def __init__(self, epsilon):
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


class GaussianNoise(object):
    """Apply Gaussian noise to an image.

    Parameters
    ----------
    mean : float
        Mean of the Gaussian noise tensor.
    std : float
        Standard deviation of the noise tensor.
    training : bool
        Only apply Gaussian noise if True.
    """
    def __init__(self, mean, std, training=True):
        self.mean = mean
        self.std = std
        self.training = training

    def __call__(self, tensor):
        if self.training:
            return gaussian_noise(tensor, self.mean, self.std)
        else:
            return tensor


def gaussian_noise(tensor, mean, stddev):
    """Apply Gaussian noise to an input tensor.

    Parameters
    ----------
    mean : float
        Mean of the Gaussian noise tensor.
    std : float
        Standard deviation of the noise tensor.
    """
    noise = Variable(tensor.data.new(tensor.size()).normal_(mean, std))
    return tensor + noise
