import numpy as np
import CostFunctions

# The structure of this neural net is as follows, where I represents an input node, N a neuron and O the output.
# I -
#     \
# I -- N - O
#     /
# I -


def train(t_inputs, t_outputs, num_iterations=10000):
    num_inputs = len(t_inputs[0])
    # we want a vector of length n, where the values are between -1 and 1
    weights = np.random.rand(1, num_inputs) * 2 - 1

    # In each iteration, we run the model on every testing input and output, and adjust the weights once
    for iteration in range(num_iterations):
        outputs = np.array([])

        for i in range(len(t_inputs)):
            input_layer = t_inputs[i]
            neuron_val = np.dot(input_layer, weights.T)
            t_output = CostFunctions.sigmoid(neuron_val)
            outputs = np.append(outputs, t_output)

        t_errors = t_outputs.T - outputs
        if iteration == num_iterations - 1:
            accuracy = 1 - np.mean(abs(t_errors))
            print("ACCURACY: \n {}".format(accuracy))

        # the magnitude of an adjustment is directly proportional to the error and the derivative of the cost function
        adjustments = np.dot(t_errors * CostFunctions.sigmoid(outputs, True), t_inputs)
        weights = weights + adjustments
    return weights


# predict(weights, input_vals) applies the trained perceptron model (weights) to the input input_vals and returns the
#   prediction as output
def predict(weights, p_input):
    neuron_val = np.dot(p_input, weights.T)
    p_output = CostFunctions.sigmoid(neuron_val)
    return p_output[0]


if __name__ == "__main__":
    training_inputs = np.array([[1, 1, 0, 0],
                                [0, 0, 1, 0],
                                [1, 0, 1, 0],
                                [0, 0, 0, 1],
                                [1, 0, 1, 1]])

    training_outputs = np.array([0, 0, 0, 1, 1])

    # notice this input is unique from the inputs trained
    input_vals = [0, 0, 0, 1]
    output = predict(train(training_inputs, training_outputs, num_iterations=10000), input_vals)
    prediction = round(output)
    error = abs(prediction - output)
    print("INPUT \n {}".format(input_vals))
    print("PREDICTION \t\t ERROR \n {} \t\t\t\t {}".format(prediction, error))
