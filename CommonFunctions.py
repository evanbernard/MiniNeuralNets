import numpy as np


# Activation Functions -------------------------------------------------------------------------------------------------
def sigmoid(x, deriv=False):
    if deriv:
        return sigmoid(x) * (1 - sigmoid(x))
    return 1 / (np.exp(-x) + 1)


def tanh(x, deriv=False):
    if deriv:
        return 1 - tanh(x, False) * tanh(x, False)
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def relu(x, deriv=False):
    if deriv:
        if x > 0:
            return 1
        return 0
    return max(0, x)


def leaky_relu(x, deriv=False, nslope=0.1):
    if deriv:
        if x > 0:
            return 1
        if x < 0:
            return nslope
        # the function derivative is not defined at x=0, but we set it to zero here to allow it to work
        return 0
    if x > 0:
        return x
    return nslope * x


def nothing(x, deriv=False):
    return x


# Error Functions ------------------------------------------------------------------------------------------------------
# mean squared error
def mse(y, y_hats, deriv=False):
    if deriv:
        return 2 * np.sum(y_hats - y)/len(y_hats)
    return np.sum((y - y_hats) ** 2) / len(y_hats)


# mean absolute error
def mae(y, y_hats, deriv=False):
    if deriv:
        if y_hats > y:
            return 1
        if y > y_hats:
            return -1
        else:
            # the derivative isn't defined when the predicted equals the actual, but in our case we don't want to
            # change the weights when this happens, so we set the output to 0
            return 0
    return np.sum(abs(y - y_hats)) / len(y_hats)
