from training.GeneticAlgorithm import genetic_algorithm
from training.StochasticGradientDescent import sgd
from training.Adam import adam
from training.BatchGradientDescent import bgd
from Perceptron import *


if __name__ == "__main__":
    x = np.array([[1, 1, 0, 0],
                  [0, 0, 1, 0],
                  [1, 0, 0, 1],
                  [0, 1, 1, 1]])
    y = np.array([0, 0, 1, 1])
    input0 = np.array([0, 1, 1, 0])

    x1 = np.array([[0],
                   [1],
                   [2],
                   [10]])
    y1 = np.array([5, 7, 9, 25])
    input1 = np.array([100])

    # Trying to model the first pattern of taking the last element in the list (so weights are [0, 0, 0, 1, 0])
    print("Genetic Algorithm ##################################################")
    nn = Perceptron(activation=leaky_relu, error=mse)
    nn.train(genetic_algorithm, x, y, 1000, 100, 0.0001)
    nn.predict(input0)

    print("Stochastic Gradient Descent ########################################")
    nn = Perceptron(activation=sigmoid, error=mse)
    nn.train(sgd, x, y, 1000, 1)
    nn.predict(input0)

    print("Batch Gradient Descent #############################################")
    nn = Perceptron(activation=sigmoid, error=mse)
    nn.train(bgd, x, y, 1000, 1)
    nn.predict(input0)

    print("Adam Optimizer #####################################################")
    nn = Perceptron(activation=sigmoid, error=mse)
    nn.train(adam, x, y, 1000, 1, 0)
    nn.predict(input0)

    # Trying to model the second pattern y = 2x + 5 (so weights are [2, 5])
    # notice the genetic algorithm cannot do this, since the weights are bound between -1 and 1.
    # Any non-momentum gradient descent performs horribly here, and the momentum based algorithms (Adam) perform
    #   extremely well, often getting exact weights [2, 5] thanks to floating point errors.
    print("Adam Optimizer #####################################################")
    nn = Perceptron(activation=leaky_relu, error=mse)
    nn.train(adam, x1, y1, 1000, 1, 0)
    nn.predict(input1)
