import numpy as np


def sigmoid(x, deriv=False):
    if deriv:
        return np.exp(-x) * sigmoid(x) * sigmoid(x)
    else:
        return 1 / (np.exp(-x) + 1)


def tanh(x, deriv=False):
    if deriv:
        return 1 - tanh(x, False) * tanh(x, False)
    else:
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
