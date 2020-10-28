from CommonFunctions import *


def bgd(activation, error_func, x, y, epochs=1000, learning_rate=0.01):
    """
    SUMMARY
        Batch Gradient Descent is the default gradient descent algorithm. The idea behind gradient descent is to adjust
        the weights in a way that slightly decreases the loss of the model, and calculates this via backpropagation.
        It computes the partial derivative of the loss function with respect to each weight (called the gradient),
        which is what indicates the direction the weight needs to be adjusted by to slightly decrease the loss. After
        each partial derivative is calculated and we have the gradient vector, we use the delta rule to evaluate the
        magnitude of the weight adjustment.
    PROS
        Since we are calculating the full gradient vector, with each weight adjustment, we know we are moving along the
        loss function in the perfect way. A common analogy is that we are a ball with no momentum rolling down a hill,
        we know that we are always going in the direction which gives us a lower cost.
    CONS
        Consider what happens when we have a large dataset. Since we compute the partial derivatives for each weight
        and only make the adjustments once after going over every data point, the algorithm can be extremely heavy on
        memory use.
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

    # In each epoch, we run the model on every weight for every testing input and output, and adjust each weight once
    for epoch in range(epochs):
        deltas = np.array([])
        for i in range(len(x)):
            input_layer = x[i]
            neuron_val = np.dot(weights, input_layer)
            y_hat = activation(neuron_val)
            # the delta rule, a generalization of the partial derivative of the cost function, thanks to backpropagation
            delta = activation(neuron_val, deriv=True) * error_func(y[i], [y_hat], deriv=True)
            deltas = np.append(deltas, delta)
        # we update the weights only after we have gone through every training data
        adjustment = learning_rate * np.dot(deltas, x)
        weights -= adjustment

    return weights, calculate_err()
