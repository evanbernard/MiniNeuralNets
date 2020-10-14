from Backpropagation import backpropagation
from GeneticAlgorithm import genetic_algorithm
from Perceptron import *


if __name__ == "__main__":
    x = np.array([[1, 1, 0, 0],
                  [0, 0, 1, 0],
                  [1, 0, 1, 0],
                  [0, 0, 0, 1],
                  [1, 0, 1, 1],
                  [0, 1, 1, 1]])

    y = np.array([0, 0, 0, 1, 1, 1])

    x1 = np.array([[-10],
                   [0],
                   [10]])
    y1 = np.array([-9, 1, 11])

    ga = Perceptron(activation=nothing, error=mse)
    ga.train(genetic_algorithm, x1, y1, 1000, 100)

    bp = Perceptron(activation=sigmoid, error=mse)
    bp.train(backpropagation, x, y, 10000)

    _input = np.array([100])
    ga.predict(_input)
    print(ga.weights)
