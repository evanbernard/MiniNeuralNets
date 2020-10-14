from CostFunctions import *
import math
import random
from operator import itemgetter


def genetic_algorithm(x, y, generations=100, num_agents=100, activation=relu, error_func=mse):
    """
    SUMMARY
        The genetic algorithm tries to mimic evolution by testing many different weights, ranking them, and making the
        weights that scored higher more likely to reproduce, where a variant of their 'genes' (weights) will be passed
        down to the next generation. After each generation, the set of weights (agents) is likely going to be slightly
        better than the previous agents, thus resulting in an improvement in accuracy over time.
    PARAMETERS
        x: a numpy array, one row for each trial (set of inputs)
        y: a numpy array, labels for the inputs x. one label per trial
        generations: integer, the number of generations to run, where once per generation the agents reproduce
        num_agents: int, the number of agents in a generation (one agent is one set of weights for the model)
        activation: function, the type of activation function to be used. (relu is the best for this model)
        error_func: function, the type of function to calculate the error. (mse is the best for this model)
    RETURN
        The function returns two elements, the numpy array of best weights found, as well as the accuracy of the
        weights, in that order.
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
            y_hats = np.array([])
            for i in range(len(x)):
                input_layer = x[i]
                neuron_val = np.dot(weights, input_layer)
                y_hat = activation(neuron_val)
                y_hats = np.append(y_hats, y_hat)
            # uses mean squared error, the error is the error of the entire trial
            er = error_func(y, y_hats)
            loe = np.append(loe, er)
        # sorts the list of weights in increasing order indexed by the error. the first element in sorted_low is
        # the best fitting model, and the last element is the worst fitting model
        sorted_tuples = sorted(tuple(zip(loe, current_low)), key=itemgetter(0))
        sorted_e = np.array([w[0] for w in sorted_tuples])
        sorted_w = np.array([w[1] for w in sorted_tuples])
        return sorted_w, sorted_e

    def selection(weights, errors):
        num_reproduced = math.ceil(len(weights)*3/4)  # we want one fourth of the population to be random - arbitrary
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
                rnd = np.random.rand(1)
                if rnd > 0.5:
                    child_weight = parent1[index]
                else:
                    child_weight = parent2[index]
                if rnd < 0.1:
                    # simulate a 10% dropout in dna while simultaneously providing random mutations
                    child_weight = 0
                child = np.append(child, child_weight)
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
