from CostFunctions import *

# The structure of this neural net is as follows, where I represents an input node, N a neuron, B a bias and O
# the output. Notice that the bias is treated simply as a node in the previous layer. The bias is extremely important
# for this model, since we will be training it with input values of either 1 or 0, so if we didn't include a bias,
# when calculating the node value we may run into the case where all inputs are 0, leading to the node value being
# sigmoid(0) = 0.5, which is no prediction at all.
# I -
#     \
# I -   N - O
#     / |
# I -   B
# The neural net is technically considered to be feed forward


class Perceptron:

    def __init__(self, weights=np.array([]), activation=sigmoid):
        self.weights = weights
        self.accuracy = 0
        self.activation = activation

    def train(self, x, y, iterations, train_func):
        self.weights, self.accuracy = train_func(x, y, iterations, self.activation)

    def predict(self, x):
        p_input = np.append(x, 1)  # add bias node
        neuron_val = np.dot(p_input, self.weights.T)
        p_output = self.activation(neuron_val)
        prediction = round(p_output[0])
        error = abs(prediction - p_output)
        return p_output[0], prediction, error[0]
