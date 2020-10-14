from CostFunctions import *


def backpropagation(x, y, iterations=1000, activation=sigmoid, error_func=difference):
    """
    SUMMARY
        The function's purpose is to generate weights that are most likely to generate a label in y, given a row of
        inputs (each row in x). It begins by randomly generating weights for the model, and it adjusts the weights by
        a factor of the derivative of the activation function and the error, so for each iteration, the weights are
        adjusted to be slightly more accurate than the previous iteration.
    PARAMETERS
        x: a numpy array, one row for each trial (set of inputs)
        y: a numpy array, labels for the inputs x. one label per trial
        iterations: integer, number of times we will adjust the weights of the model
        activation: function, the type of activation function to be used. (sigmoid is the best for this model)
        error_func: function, the type of function to calculate the error. (difference is the best for this model)
    RETURN
        The function returns two elements, the numpy array of best weights found, as well as the accuracy of the
        weights, in that order.
    """
    num_inputs = len(x[0])
    accuracy = 0
    # we are adding a bias by creating a new node (val=1) in the input layer and treating it as an input, so append
    #   1 to the end of each row in the inputs. Doing it this way is easier since we can adjust the weight of the
    #   bias along with the rest of the nodes in the previous layer
    temp = []
    for row in range(len(x)):
        temp.append(np.append(x[row], 1))
    x = np.array(temp)

    # we want a vector of length n + 1 (one extra for a bias) for weights, where the values are between -1 and 1
    weights = np.random.rand(1, num_inputs + 1) * 2 - 1

    # In each iteration, we run the model on every testing input and output, and adjust the weights once
    for iteration in range(iterations):
        y_hats = np.array([])
        neuron_vals = np.array([])

        for i in range(len(x)):
            input_layer = x[i]
            neuron_val = np.dot(weights, input_layer)
            y_hat = activation(neuron_val)
            y_hats = np.append(y_hats, y_hat)
            neuron_vals = np.append(neuron_vals, neuron_val)

        t_errors = error_func(y, y_hats) * activation(neuron_vals, deriv=True)

        if iteration == iterations - 1:
            accuracy = 1 - np.mean(abs(t_errors))

        # the magnitude of an adjustment is directly proportional to the error, we dot product errors with the input
        #   in order to remove the affect the error from an input of 0 has. When the input is 0, that input does not
        #   affect the value of the node, so it's error is irrelevant. See the readme on github for more detail
        adjustments = np.dot(t_errors, x)

        weights = weights + adjustments
    return weights, accuracy
