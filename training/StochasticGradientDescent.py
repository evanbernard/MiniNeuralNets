from CommonFunctions import *


def sgd(activation, error_func, x, y, epochs=1000, learning_rate=0.01):
    """
    SUMMARY
        Stochastic Gradient Descent is a variant of regular gradient descent (batch gradient descent, or bgd), and it
        is effectively the same algorithm, with the exception of when the weights update. Recall that bgd updates all
        of the weights at the same time, after the full gradient vector is calculated. What sgd does instead is update
        each weight as soon as it's partial derivative is calculated.
    PROS
        Since the weights are updated very frequently, the loss may fluctuate very heavily, allowing it to possibly
        move to a new, lower local minimum. Sgd is very low on memory compared to bgd (imagine storing hundreds of
        thousands of gradients in memory before finally adjusting the weight values)
    CONS
        The update frequency of the weights is also a downside, since it often results in a very choppy path to the
        minimum of the cost function. It's sort of like a drunk man trying to walk down a hill, where bgd is a ball
        with no momentum; it always adjusts it's direction to be follow the exact path of the hill.
    ARGUMENTS
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

    def calculate_err():
        y_hats = np.array([])
        for u in range(len(x)):
            y_hats = np.append(y_hats, activation(np.dot(weights, x[u])))
        return error_func(y, y_hats)

    num_inputs = len(x[0])

    # we are adding a bias by creating a new node (val=1) in the input layer and treating it as an input
    temp = []
    for row in range(len(x)):
        temp.append(np.append(x[row], 1))
    x = np.array(temp)

    # we want a vector of length n + 1 (one extra for a bias) for weights, where the values are between -1 and 1
    weights = (np.random.rand(1, num_inputs + 1) * 2 - 1)[0]
    tup = list(zip(x, y))
    # In each epoch, we run the model on every weight for every testing input and output, and adjust each weight once
    for epoch in range(epochs):
        # since we update every weight as soon as we calculate the gradient, we need to randomize it every epoch
        np.random.shuffle(tup)
        x = [x1 for x1, _ in tup]
        y = [x2 for _, x2 in tup]
        for i in range(len(x)):
            input_layer = x[i]
            neuron_val = np.dot(weights, input_layer)
            y_hat = activation(neuron_val)
            # the delta rule, a generalization of the partial derivative of the cost function, thanks to backpropagation
            delta = activation(neuron_val, deriv=True) * error_func(y[i], y_hat, deriv=True)
            for k in range(num_inputs + 1):
                adjustment = delta * learning_rate * x[i][k]
                # we update each individual weight once per trial
                weights[k] -= adjustment

    return weights, calculate_err()
