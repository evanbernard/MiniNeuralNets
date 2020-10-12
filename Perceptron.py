from CostFunctions import *

# The structure of this neural net is as follows, where I represents an input node, N a neuron and O the output.
# I -
#     \
# I -- N - O
#     /
# I -
# The neural net is trained via backpropagation


def train(t_inputs, t_outputs, num_iterations=1000):
    num_inputs = len(t_inputs[0])
    # we want a vector of length n, where the values are between -1 and 1
    weights = np.random.rand(1, num_inputs) * 2 - 1

    # In each iteration, we run the model on every testing input and output, and adjust the weights once
    for iteration in range(num_iterations):
        outputs = np.array([])

        for i in range(len(t_inputs)):
            input_layer = t_inputs[i]
            neuron_val = np.dot(input_layer, weights.T)
            output = sigmoid(neuron_val)
            outputs = np.append(outputs, output)

        t_errors = (t_outputs.T - outputs) * (outputs * (1 - outputs))

        if iteration == num_iterations - 1:
            accuracy = 1 - np.mean(abs(t_errors))
            print("ACCURACY: \n {}".format(accuracy))
        # the magnitude of an adjustment is directly proportional to the error
        adjustments = np.dot(t_errors, t_inputs)
        weights = weights + adjustments
    return weights


# predict(weights, input_vals) applies the trained perceptron model (weights) to the input input_vals and returns the
#   prediction as output
def predict(weights, p_input):
    neuron_val = np.dot(p_input, weights.T)
    p_output = sigmoid(neuron_val)
    return p_output[0]


if __name__ == "__main__":
    training_inputs = np.array([[1, 1, 0, 0],
                                [0, 0, 1, 0],
                                [1, 0, 1, 0],
                                [0, 0, 0, 1],
                                [1, 0, 1, 1],
                                [0, 1, 1, 1]])

    training_outputs = np.array([0, 0, 0, 1, 1, 1])

    # notice this input is unique from the inputs trained
    input_vals = [0, 1, 1, 0]
    output = predict(train(training_inputs, training_outputs, num_iterations=10000), input_vals)
    prediction = round(output)
    error = (prediction - output)**2
    print("INPUT \n {}".format(input_vals))
    print("PREDICTION \t\t ERROR \n {} \t\t\t {}".format(prediction, error))
