import numpy as np


# sigmoid(x, deriv) takes in a number (x) and boolean (deriv), and will return the result of the sigmoid function
#   applied to x if deriv is False, and it will return the result of the derivative of the sigmoid function
#   otherwise
def sigmoid(x, deriv=False):
    if deriv:
        return np.exp(-x) / (np.exp(-x) + 1) ** 2
    else:
        return 1 / (np.exp(-x) + 1)
