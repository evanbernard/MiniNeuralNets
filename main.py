from Backpropagation import backpropagation
from GeneticAlgorithm import genetic_algorithm
from Perceptron import *


if __name__ == "__main__":
    x = np.array([[1, 1, 0, 0],
                  [0, 0, 1, 0],
                  [1, 0, 1, 1],
                  [0, 1, 1, 1]])

    y = np.array([0, 0, 1, 1])

    x1 = np.array([[-2],
                   [-1],
                   [0],
                   [1],
                   [2]])
    y1 = np.array([-1, 0, 1, 2, 3])

    ga = Perceptron(activation=relu, error=mse)
    ga.train(genetic_algorithm, x1, y1, 1000, 100)

    # bp = Perceptron(activation=sigmoid, error=mse)
    # bp.train(backpropagation, x, y, 10000, 1)

    _input = np.array([5])
    ga.predict(_input)
