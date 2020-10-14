from CostFunctions import *
import math
import random
from operator import itemgetter


def genetic_algorithm(x, y, generations=1000, num_agents=100, activation=sigmoid):
    """ The genetic algorithm tries to mimic evolution by trying many different weights, ranking them, and allowing
        the highest-ranking weights to reproduce and have their 'genes' make it to the next generation.

    :param x: numpy array representing the training data, a matrix of inputs where each row is one trial
    :param y: numpy array, the labels for the training data
    :param generations: int, the number of generations to run
    :param activation: function, the activation function to be applied on the node
    :param num_agents: int, the number of weights in a generation
    :return: the weights and accuracy of the trained model, as a numpy array and float respectively
    """

    def generate_agents(n):
        # create random list of weights, it has an extra value for the bias weight
        weights = np.empty((0, num_inputs + 1))
        for i in range(n):
            new_weight = np.random.rand(1, num_inputs + 1) * 2 - 1
            weights = np.append(weights, new_weight, axis=0)
        return weights

    def calculate_fitness(current_low):
        loe = np.empty((0, num_inputs + 1)),
        for weights in current_low:
            outputs = np.array([])
            for i in range(len(x)):
                input_layer = x[i]
                neuron_val = np.dot(weights, input_layer)
                y_hat = activation(neuron_val)
                outputs = np.append(outputs, y_hat)
            # uses mean squared error, the error is the error of the entire trial
            er = np.sum((y - outputs)**2)/len(outputs)
            loe = np.append(loe, er)
        # sorts the list of weights in increasing order indexed by the error. the first element in sorted_low is
        # the best fitting model, and the last element is the worst fitting model
        sorted_tuples = sorted(tuple(zip(loe, current_low)), key=itemgetter(0))
        sorted_e = np.array([w[0] for w in sorted_tuples])
        sorted_w = np.array([w[1] for w in sorted_tuples])
        return sorted_w, sorted_e

    def selection(weights, errors):
        num_reproduced = math.ceil(len(weights)*3/4)  # we want three fourths of the population to reproduce
        num_new = num_agents - num_reproduced  # the rest of the agents will be generated randomly

        # we want low error to have a large probability, so divide errors by 1
        error_sum = np.sum(np.divide(1, errors))
        count = 0
        probabilities = []
        for i in range(len(weights)):
            if i == len(weights) - 1:
                # ensure sum of probabilities is equal to 1
                num = 1 - count
            else:
                # divide by the sum of errors so the sum is equal to 1 (without floating-point errors)
                num = (1/errors[i])/error_sum
            count += num
            probabilities.append(num)

        index_choices = np.random.choice(len(weights), num_reproduced, p=probabilities)
        weights = weights[index_choices]

        new_weights = []
        # simulate dna swapping during reproduction
        for i in range(num_reproduced):
            parent1 = random.choice(weights)
            parent2 = random.choice(weights)
            child = np.array([])
            for index in range(len(parent1)):
                child = np.append(child, (parent1[index] + parent2[index])/2)
            new_weights.append(child)

        new_weights = np.array(new_weights)
        random_weights = generate_agents(num_new)
        new_weights = np.append(new_weights, random_weights, axis=0)
        return new_weights

    num_inputs = len(x[0])
    temp = []
    for row in range(len(x)):
        temp.append(np.append(x[row], 1))
    x = np.array(temp)

    low = generate_agents(num_agents)

    for generation in range(generations):
        sorted_weights, sorted_errors = calculate_fitness(low)
        low = selection(sorted_weights, sorted_errors)

    best_weight, error = calculate_fitness(low)
    return best_weight[0], error[0]
