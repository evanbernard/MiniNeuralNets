# MiniNeuralNets
A framework for mini neural networks (single node networks, also known as perceptrons), written from scratch in python.

## main.py
This script acts as the controller for the mini neural networks. You're able to instantiate a perceptron with
```python
nn = Perceptron(activation=sigmoid, error=mse)
```
This will store a perceptron in nn, which contains the weights, activation function, error function and accuracy for the model. To train the model, you call `nn.train`, passing the training algorithm of your choice, along with the arguments your chosen training algorithm needs. As an example, to train a model using backpropagation on inputs `x`, labels `y`, with 10000 iterations, you pass the following parameters:
```python
nn.train(backpropagation, x, y, 10000)
```
Then, to make a prediction of some input `input` using the trained model, you call
```python
output, prediction, confidence = nn.predict(input)
```
where `output` is the actual output of the model, `prediction` is the rounded `output`, and `confidence` is the confidence the model has for it's prediction.

## Backpropagation.py
The backpropagation algorithm is a method of using gradient descent to adjust the weights of the model in an attempt to minimize the error function. The idea is to adjust the weights significantly when the error is large, and to make progressively smaller adjustments the closer the error is to 0. The 'gradient' in 'gradient descent' comes from the algorithm's use of the gradient of the error function to determine the direction of the adjustment. For each weight, we need to calculate the delta value, which is found by multiplying the derivative of the activation function of the neuron by the difference between the target output and produced output. Then, to calculate the adjustment amount for that weight, we multiply the delta value by the input value attached to the weight, and we are left with the amount we need to add to the weight in order to slightly improve the accuracy of the model. Notice the derivation is dependent on the gradient of the error function, so the script only has functionality to perform the operation using the traditional mean squared error function.

We will now train a model using backpropagation and sigmoid to predict the output of the following pattern:
```python 
x = np.array([[1, 1, 0, 0],
              [0, 0, 1, 0],
              [1, 0, 1, 1],
              [0, 1, 1, 1]])

y = np.array([0, 0, 1, 1])

nn = Perceptron(activation=sigmoid)
# x are the training inputs, y are training labels, 10000 iterations and a learning rate of 1
nn.train(backpropagation, x, y, 10000, 1)

input_vals = np.array([0,1,1,0])
nn.predict(input_vals)
```
Notice the output is simply the last element in the input. It's difficult for a perceptron to find more complex patterns, after all, it's only one node. The output is as follows:
```python
# OUTPUT
INPUT 
 [0 1 1 0]
PREDICTION 	 CONFIDENCE 
 0.0 		 0.9905242286545263
WEIGHTS
 [-0.07335839 -0.07307256  0.20623714  9.57783444 -4.78266075]
```
So, we have trained a perceptron to model the given pattern, and it has produced the correct result with a confidence of 99.05%, on an input it has never seen before. Notice that each weight corresponds to it's respective input, i.e., the first value in the input pattern has a weight of -0.0734, the last value in the input pattern has a weight of 9.578, and the bias has a weight of -4.783. From these weights, it's easy to see how the model calculates the result given an input, it essentially ignores the first 3 inputs, making the value of the neuron entirely dependent on the last input value, which is exactly how it is expected to model the pattern.

## GeneticAlgorithm.py
The genetic algorithm tries to mimic evolution, by adjusting the weights of a given model until it's found optimal weights. It's typically used in a unsupervised way, measuring the performance of the model based on the 'score' it obtains in an environment it's being trained in. However, since this is a learning exercise, we use the algorithm in a supervised way, calculating the score of the model based on how close the output is to the actual answer. There are 5 main steps to this algorithm:
1. Generate Agents
We randomly initialize `num_agents` agents (weights in the model). One agent is the equivalent of one set of weights for the model.
2. Measure Fitness
We find some way to calculate the fitness of each agent. In our case, since we have the desired outputs, we use the error of the model's prediction (using a user-passed error function) to rank the fitnesses of the agents.
3. Selection
We choose some number of agents to reproduce, where the higher the agent's fitness, the more likely they are to be selected.
4. Reproduction
Out of the pool of selected agents, we randomly pick two to reproduce, until we have close to the number of agents we started the algorithm with. The reproduction is the creation of a child, where the child's weights is taken to be some combination of the parent's weights. Each weight in the child is chosen to be one of the two parent's weight in that index, and there is a 10% chance for any given weight to be set to 0. The random mutations allow for new patterns to emerge (in case the model gets stuck in a local minimum), and it also simulates the idea of dropout, which again prevents the model from being stuck in a local minimum, and allows it to explore new patterns.
5. Create New Generation
When we have a number of children that are close to the number of agents we started with (in our case, it's 75%, chosen fairly arbitrarily), we fill the rest of the generation with randomized weights, to again allow for new patterns to emerge.

This process is performed `generations` times, and the assumption of the algorithm is that with each new generation, the weights of the agents have slightly improved, because the stronger performing weights were able to reproduce. We will now give an example of the genetic algorithm in play, with a very simple pattern.

```python
x = np.array([[-2],
              [-1],
              [0],
              [1],
              [2]])
y = np.array([-1, 0, 1, 2, 3])

nn = Perceptron(activation=relu, error=mse)
# we will simulate 1000 generations, with each having 100 agents
nn.train(genetic_algorithm, x, y, 1000, 100)

input_vals = np.array([10])
nn.predict(input_vals)
```
Clearly the pattern returns `x+1`, given some `x`. Notice we are using the ReLU activation function. This function takes in some `x` and simply returns the maximum of `x` and 0. Since we want our output to be an unbouded integer, we are restricted from using activation functions such as sigmoid or tanh, which is why we choose ReLU. The output is as follows:
```python
# OUTPUT
INPUT 
 [10]
PREDICTION 	 CONFIDENCE 
 11.0 		 0.9342778404837393
WEIGHTS
 [0.99634235 0.97085438]
```
So, we've successfully trained the perceptron to model the function `x+1`. Consider the weights, noticing that the first weight is the connection between the input and the node value (which is the output, since this is a perceptron), and the second weight is the connection between the bias (value=1) and the node value. It isn't hard to see that if the model was 100% accurate, the weights would be `[1, 1]`, since to calculate the node value, you add the result of the first weight multiplied by the input and the second weight mulitplied by the bias (value=1). When both weights are `1`, this simplifies to `node_value = input+1`, which is exactly our desired relationship.
