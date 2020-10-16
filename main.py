from TrainingAlgorithms.GeneticAlgorithm import genetic_algorithm
from TrainingAlgorithms.GradientDescent import gradient_descent
from Perceptron import *


if __name__ == "__main__":
    x = np.array([[1, 1, 0, 0],
                  [0, 0, 1, 0],
                  [1, 0, 0, 1],
                  [0, 1, 1, 1]])

    y = np.array([0, 0, 1, 1])

    x1 = np.array([[0],
                   [1],
                   [2]])
    y1 = np.array([1, 2, 3])

    ga = Perceptron(activation=relu, error=mae)
    ga.train(genetic_algorithm, x1, y1, 1000, 100, 0.002)

    gd = Perceptron(activation=sigmoid, error=mse)
    gd.train(gradient_descent, x, y, 10000, 1)

    _input = np.array([10])
    ga.predict(_input)

    _input = np.array([0, 1, 1, 0])
    gd.predict(_input)
