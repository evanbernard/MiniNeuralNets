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

    def __init__(self, weights=np.array([]), activation=sigmoid, error=mse):
        self.weights = weights
        self.accuracy = 0
        self.activation = activation
        self.error = error

    def train(self, train_func, *args):
        self.weights, self.accuracy = train_func(*args, self.activation, self.error)

    def predict(self, x, display=True):
        p_input = np.append(x, 1)  # add bias node
        neuron_val = np.dot(self.weights, p_input)
        p_output = self.activation(neuron_val)
        # depending on the activation function the output may be a numpy array, so force it to be a value
        if type(p_output) is np.ndarray:
            p_output = p_output[0]
        prediction = round(p_output)
        confidence = max(abs(prediction - p_output), 1 - abs(prediction - p_output))

        if display:
            print("INPUT \n {}".format(x))
            print("PREDICTION \t CONFIDENCE \n {} \t\t {}".format(prediction, confidence))
            print("NEURON VAL \n {}".format(neuron_val))
        return p_output, prediction, confidence
