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
# The neural net is considered to be feed forward and it is trained via backpropagation


def train(t_inputs, t_outputs, num_iterations=1000):
    num_inputs = len(t_inputs[0])
    # we are adding a bias by creating a new node (val=1) in the previous layer and treating it as an input, so append
    #   1 to the end of each row in the inputs. Doing it this way is easier since we can adjust the weight of the
    #   bias along with the rest of the nodes in the previous layer
    temp = []
    for row in range(len(t_inputs)):
        temp.append(np.append(t_inputs[row], 1))
    t_inputs = np.array(temp)

    # we want a vector of length n + 1 (one extra for a bias) for weights, where the values are between -1 and 1
    weights = np.random.rand(1, num_inputs + 1) * 2 - 1

    # In each iteration, we run the model on every testing input and output, and adjust the weights once
    for iteration in range(num_iterations):
        outputs = np.array([])

        for i in range(len(t_inputs)):
            input_layer = t_inputs[i]
            neuron_val = np.dot(input_layer, weights.T)
            y_hat = sigmoid(neuron_val)
            outputs = np.append(outputs, y_hat)

        t_errors = (t_outputs.T - outputs) * (outputs * (1 - outputs))

        if iteration == num_iterations - 1:
            accuracy = 1 - np.mean(abs(t_errors))
            print("ACCURACY: \n {}".format(accuracy))
        # the magnitude of an adjustment is directly proportional to the error
        adjustments = np.dot(t_errors, t_inputs)
        weights = weights + adjustments
    return weights


# predict(p_weights, p__vals) applies the trained perceptron model (weights) to the input input_vals and returns the
#   prediction as output
def predict(p_weights, p_input):
    # add a 1 to the end of the inputs for the bias node
    p_input = np.append(p_input, 1)
    neuron_val = np.dot(p_input, p_weights.T)
    p_output = sigmoid(neuron_val)
    prediction = round(p_output[0])
    error = (prediction - p_output) ** 2
    return p_output[0], prediction, error[0]


if __name__ == "__main__":
    training_inputs = np.array([[1, 1, 0, 0],
                                [0, 0, 1, 0],
                                [1, 0, 1, 0],
                                [0, 0, 0, 1],
                                [1, 0, 1, 1],
                                [0, 1, 1, 1]])
    training_outputs = np.array([0, 0, 0, 1, 1, 1])

    weights = train(training_inputs, training_outputs, num_iterations=10000)

    input_vals = np.array([0, 0, 0, 0])
    _, prediction, error = predict(weights, input_vals)
    print("INPUT \n {}".format(input_vals))
    print("PREDICTION \t\t ERROR \n {} \t\t\t {}".format(prediction, error))
