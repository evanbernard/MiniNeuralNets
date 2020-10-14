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

    ga = Perceptron(activation=relu, error=mse)
    ga.train(genetic_algorithm, x, y, 100, 50)

    bp = Perceptron(activation=sigmoid, error=difference)
    bp.train(backpropagation, x, y, 100)

    input_vals = np.array([0, 1, 1, 0])
    bp.predict(input_vals)
    ga.predict(input_vals)
