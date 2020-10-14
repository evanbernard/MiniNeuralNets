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


def relu(x):
    return max(0, x)


def nothing(x):
    return x


def mse(y, y_hats):
    return np.sum((y - y_hats) ** 2) / len(y_hats)
