from CostFunctions import *


def backpropagation(x, y, num_iterations=1000, activation=sigmoid):
    num_inputs = len(x[0])
    accuracy = 0
    # we are adding a bias by creating a new node (val=1) in the previous layer and treating it as an input, so append
    #   1 to the end of each row in the inputs. Doing it this way is easier since we can adjust the weight of the
    #   bias along with the rest of the nodes in the previous layer
    temp = []
    for row in range(len(x)):
        temp.append(np.append(x[row], 1))
    t_inputs = np.array(temp)

    # we want a vector of length n + 1 (one extra for a bias) for weights, where the values are between -1 and 1
    weights = np.random.rand(1, num_inputs + 1) * 2 - 1

    # In each iteration, we run the model on every testing input and output, and adjust the weights once
    for iteration in range(num_iterations):
        outputs = np.array([])
        neuron_vals = np.array([])

        for i in range(len(t_inputs)):
            input_layer = t_inputs[i]
            neuron_val = np.dot(input_layer, weights.T)
            y_hat = activation(neuron_val)
            outputs = np.append(outputs, y_hat)
            neuron_vals = np.append(neuron_vals, neuron_val)

        t_errors = (y - outputs) * activation(neuron_vals, deriv=True)

        if iteration == num_iterations - 1:
            accuracy = 1 - np.mean(abs(t_errors))

        # the magnitude of an adjustment is directly proportional to the error
        adjustments = np.dot(t_errors, t_inputs)
        weights = weights + adjustments
    return weights, accuracy
