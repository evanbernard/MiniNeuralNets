from CommonFunctions import *


def gradient_descent(x, y, epochs=1000, learning_rate=1, activation=sigmoid, error_func=mse):
    """
    SUMMARY
        The function's purpose is to adjust the neural network's weights in order to accurately produce the some label
        y[i] given a row of inputs x[i]. It begins by randomly generating weights for the model, and it adjusts the
        weights by a factor of the derivative of the activation function and the error, so for each epoch, the weights
        are adjusted to be slightly more accurate than the previous iteration.
    PARAMETERS
        x: a numpy array, one row for each trial (set of inputs)
        y: a numpy array, labels for the inputs x. one label per trial
        epochs: integer, number of times we will pass through all training data the weights of the model
        learning_rate: float, a relative value which determines how big the weight adjustments are. Too high and the
            model won't be able to find the local minimum, too small and the model will take too long to find the
            minimum.
        activation: function, the type of activation function to be used. (sigmoid works well)
        error_func: function, the type of error function to be used. (mean absolute error works well)
    RETURN
        The function returns two elements, the numpy array of best weights found, as well as the error of the
        weights, in that order.
    NOTE
        The activation function and error functions should be differentiable, because the delta values are dependent
        on the derivatives of those functions.
    """
    num_inputs = len(x[0])
    er = 1
    # we are adding a bias by creating a new node (val=1) in the input layer and treating it as an input, so append
    #   1 to the end of each row in the inputs. Doing it this way is easier since we can adjust the weight of the
    #   bias along with the rest of the nodes in the previous layer
    temp = []
    for row in range(len(x)):
        temp.append(np.append(x[row], 1))
    x = np.array(temp)

    # we want a vector of length n + 1 (one extra for a bias) for weights, where the values are between -1 and 1
    weights = (np.random.rand(1, num_inputs + 1) * 2 - 1)[0]

    # In each epoch, we run the model on every weight for every testing input and output, and adjust each weight once
    for epoch in range(epochs):
        for i in range(len(x)):
            input_layer = x[i]
            neuron_val = np.dot(weights, input_layer)
            y_hat = activation(neuron_val)
            # the delta rule, a generalization of the partial derivative of the cost function, thanks to backpropagation
            delta = activation(neuron_val, deriv=True) * error_func(y[i], [y_hat], deriv=True)
            for k in range(num_inputs+1):
                adjustment = delta * learning_rate * x[i][k]
                # we update each individual weight once per trial
                weights[k] -= adjustment

        # calculate error on the last iteration
        if epoch == epochs - 1:
            y_hats = np.array([])
            for i in range(len(x)):
                # same calculation as above, just written on one line
                y_hats = np.append(y_hats, activation(np.dot(weights, x[i])))
            er = error_func(y, y_hats)

    return weights, er
