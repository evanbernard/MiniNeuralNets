from CommonFunctions import *
import math
import random
from operator import itemgetter


def genetic_algorithm(activation, error_func, x, y, generations=100, num_agents=100, stop_error=0.001):
    """
    SUMMARY
        The genetic algorithm tries to mimic evolution by testing many different weights, ranking them, and making the
        weights that scored higher more likely to reproduce, where a variant of their 'genes' (weights) will be passed
        down to the next generation. After each generation, the set of weights (agents) is likely going to be slightly
        better than the previous agents, thus resulting in an improvement in accuracy over time.
    PROS
        The algorithm is (typically) unsupervised, meaning it needs no training labels to improve overtime. The
        algorithm excels when used on an agent in a video game, since you're able to calculate the fitness of the agent
        based on how well it performs in the environment it's in. It also is able to get itself out of local minimums
        fairly easily, thanks to random genetic mutations.
    CONS
        The algorithm is very computationally expensive.
    ARGUMENTS
        activation: function, the type of activation function to be used. (relu performs well with this model)
        error_func: function, the type of function to calculate the error. (mae performs well with this model)
        x: a numpy array, one row for each trial (set of inputs)
        y: a numpy array, labels for the inputs x. one label per trial
        generations: integer, the number of generations to run, where once per generation the agents reproduce
        num_agents: int, the number of agents in a generation (one agent is one set of weights for the model)
        stop_error: float, if the error of the model is under or equal to this value, the model will be returned as it
            is. It's very useful for the genetic algorithm due to the randomized nature of the algorithm; it's possible
            to find a well fitting model, but the random mutations alter the model before the accurate model is returned
    RETURN
        The function returns two elements, the numpy array of best weights found, as well as the error of the
        weights, in that order.
    NOTE
        This algorithm is almost always used for unsupervised training, i.e. in situations where the actual result
        you want your model to predict is unknown. This algorithm is common when training an agent to play a video game,
        since you aren't able to know what move is the optimal move. However, since this is a learning exercise, we
        will use the algorithm in a supervised way, by calculating the 'fitness' of the model directly, using the error
        of the prediction.
    """

    def generate_agents(n):
        # create random list of weights, it has an extra value for the bias weight
        weights = np.empty((0, num_inputs + 1))
        for i in range(n):
            new_weight = np.random.rand(1, num_inputs + 1) * 2 - 1
            weights = np.append(weights, new_weight, axis=0)
        return weights

    def calculate_fitness(current_gen):
        loe = np.empty((0, num_inputs + 1)),
        for weights in current_gen:
            y_hats = np.array([])
            for i in range(len(x)):
                input_layer = x[i]
                neuron_val = np.dot(weights, input_layer)
                y_hat = activation(neuron_val)
                y_hats = np.append(y_hats, y_hat)
            # the error of the agent
            er = error_func(y, y_hats)
            if er < stop_error:
                print("EARLY STOP")
                return weights, er, True

            loe = np.append(loe, er)
        # sorts the list of weights in increasing order indexed by the error. the first element in sorted_low is
        # the best fitting model, and the last element is the worst fitting model
        sorted_tuples = sorted(tuple(zip(loe, current_gen)), key=itemgetter(0))
        sorted_e = np.array([w[0] for w in sorted_tuples])
        sorted_w = np.array([w[1] for w in sorted_tuples])
        return sorted_w, sorted_e, False

    def selection(weights, errors):
        num_reproduced = math.ceil(len(weights)*3/4)  # we want one fourth of the population to be random, arbitrary %
        num_new = num_agents - num_reproduced  # the rest of the agents will be generated randomly

        # we want low errors to have a large probability of being selected, so divide errors by 1
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
                    # simulate 10% random mutations
                    child_weight = np.random.rand(1) * 2 - 1
                child = np.append(child, child_weight)
            new_weights.append(child)

        new_weights = np.array(new_weights)
        random_weights = generate_agents(num_new)
        new_weights = np.append(new_weights, random_weights, axis=0)
        return new_weights

    num_inputs = len(x[0])
    # add a bias node to the input layer, with a value of 1
    temp = []
    for row in range(len(x)):
        temp.append(np.append(x[row], 1))
    x = np.array(temp)

    # initial list of weights (a generation)
    low = generate_agents(num_agents)

    for generation in range(generations):
        sorted_weights, sorted_errors, early_quit = calculate_fitness(low)
        if early_quit:
            return sorted_weights, sorted_errors
        low = selection(sorted_weights, sorted_errors)

    best_weight, error, _ = calculate_fitness(low)
    return best_weight[0], error[0]
