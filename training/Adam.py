from CommonFunctions import *


def adam(activation, error_func, x, y, epochs=1000, learning_rate=0.001, stop_loss=0.001):
    """
    SUMMARY
        Using https://arxiv.org/abs/1412.6980 as a reference.
        Adam is an optimization of gradient descent. The algorithm is essentially a combination of Stochastic Gradient
        Descent with momentum and RMSprop. Adam adjusts the learning rate for each weight (RMSprop), and uses the
        moving average of the gradient rather than the gradient itself (momentum). This algorithm can be thought of as
        a ball with lots of friction rolling down a hill. It's often the default optimization algorithm used when
        training neural networks.
    PROS
        Adam combines the best of RMSprop and sgd with momentum to have a computationally efficient and scalable
        algorithm, with hyperparameters that require no tuning the large majority of the time.
    CONS
        May suffer weight decay problems (the weights go to zero, and are unable to return).
    ARGUMENTS
        x: a numpy array, one row for each trial (set of inputs)
        y: a numpy array, labels for the inputs x. one label per trial
        epochs: integer, number of times we will pass through all training data the weights of the model
        learning_rate: float, a relative value which determines how big the weight adjustments are. Too high and the
            model won't be able to find the local minimum, too small and the model will take too long to find the
            minimum.
        stop_loss: float, if every weight is being adjusted by a value is smaller than stop_loss, the current weights
            are deemed acceptable, and are returned
        activation: function, the type of activation function to be used. (sigmoid works well)
        error_func: function, the type of error function to be used. (mean squared error works well)
    RETURN
        The function returns two elements, the numpy array of best weights found, as well as the error of the
        weights, in that order.
    NOTE
        Typically Adam performs the weight adjustments on a random mini-batch of data (like mini-batch gradient
        descent), however, since we manually define the training dataset, we have a very small dataset, so we are able
        to treat our entire dataset as a mini-batch.
    """

    num_inputs = len(x[0])
    # we are adding a bias by creating a new node (val=1) in the input layer and treating it as an input
    temp = []
    for row in range(len(x)):
        temp.append(np.append(x[row], 1))
    x = np.array(temp)
    # the two moments, mean and variance
    m = np.zeros(num_inputs + 1)
    v = np.zeros(num_inputs + 1)
    # m moment's exponential decay rate
    beta_1 = 0.9
    # v moment's exponential decay rate
    beta_2 = 0.999
    # prevent division by 0
    epsilon = 10**-8

    # we want a vector of length n + 1 (one extra for a bias) for weights, where the values are between -1 and 1
    weights = (np.random.rand(1, num_inputs + 1) * 2 - 1)[0]

    def calculate_err():
        y_hats = np.array([])
        for k in range(len(x)):
            y_hats = np.append(y_hats, activation(np.dot(weights, x[k])))
        return error_func(y, y_hats)

    # In each epoch, we run the model on every weight for every testing input and output, and adjust each weight once
    for epoch in range(epochs):
        t = epoch + 1  # prevent division by 0
        for i in range(len(x)):
            input_layer = x[i]
            neuron_val = np.dot(weights, input_layer)
            y_hat = activation(neuron_val)

            # g is the partial derivative of the cost w.r.t the weight, just like in gradient descent
            g = (activation(neuron_val, deriv=True) + epsilon) * error_func(y[i], [y_hat], deriv=True) * input_layer

            # adjust mean and variance such that the decay rate decreases them over time
            m = beta_1 * m + (1 - beta_1) * g
            v = beta_2 * v + (1 - beta_2) * np.power(g, 2)

            m_hat = m / (1 - np.power(beta_1, t))
            v_hat = v / (1 - np.power(beta_2, t))

            # final adjustment value, notice epsilon added purely to avoid division by 0
            adjustment = learning_rate * m_hat/(np.sqrt(v_hat) + epsilon)

            if all(abs(x) < stop_loss for x in adjustment):
                print("EARLY STOP")
                return weights, calculate_err()

            weights = weights - adjustment

    return weights, calculate_err()
