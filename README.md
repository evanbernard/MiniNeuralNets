# MiniNeuralNets
A framework for mini neural networks (single node networks, also known as perceptrons), written from scratch in python.

## main.py
This script acts as the controller for the mini neural networks. You're able to instantiate a perceptron with
```python
nn = Perceptron(activation=sigmoid)
```
This will store a perceptron in nn, which contains the weights, activation function and accuracy for the model. To train the model, you call `nn.train` passing the training inputs `x` and training labels `y`, along with the number of iterations and the training algorithm you'd like to use. We will train the model using backpropagation and sigmoid to predict the output of the following pattern:
```python
x = np.array([[1, 1, 0, 0],
              [0, 0, 1, 0],
              [1, 0, 1, 0],
              [0, 0, 0, 1],
              [1, 0, 1, 1],
              [0, 1, 1, 1]])
y = np.array([0, 0, 0, 1, 1, 1])
nn.train(x, y, iterations=10000, train_func=backpropagation)
```
Notice the output is simply the last element in the input, it's difficult for a perceptron to find more complex patterns, after all, it's only one node. Now that you have a trained perceptron, you can make predictions with it using `nn.predict(x)`, for some input `x`. The method returns the pre-activation output, the prediction (rounded activation function on the output), as well as the confidence of the choice repsectively.
```python
input_vals = np.array([0, 1, 1, 0])
output, prediction, condidence = nn.predict(input_vals)
```
```python
# OUTPUT
INPUT 
 [0 1 1 0]
PREDICTION 	 CONFIDENCE 
 0.0 		 0.9945485299594722
```
As you can see, we've successfully trained our perceptron to model the pattern given, and it was able to predict the result of an input it's never seen before, with a confidence of 99.5%.

## Backpropagation.py
Backpropagation is a method of adjusting the weights in a neural network. The central idea behind this algorithm is to significantly adjust the weights when the error is large, and make smaller adjustments to the weights when the error is small, where the direction of the adjustment is relative to the gradient of the activation function. This can be done by setting the weight adjustment values to be the vector of errors multiplied by the matrix of inputs. The vector of errors is calculated element-wise by multiplying the difference between the output and the actual answer by the derivative of the activation function of our choosing. The result is the adjustment vector, where the `ith` element is the amount that the `ith` node's weight should change, and so we complete the backpropagation algorithm by adding the adjustment vector to the weights, leaving us with slightly more accurate weights. The calculation of the weight adjustment vector can be seen here:
```
errors = [e1 e2 e3 e4 e5 e6]      

         [1 1 0 0]
         [0 0 1 0]
inputs = [1 0 1 0]
         [0 0 0 1]
         [1 0 1 1]
         [0 1 1 1]
         
adjustments = errors * inputs
adjustments[0] == e1 + e3 + e5
```
Recall that the first column in the inputs matrix would be the first nodes in the trials (one row of inputs is one trial), and `ei` is the error for the `ith` input. So, `adjustments[0]` is the dot product of the errors from each trial with the first node input of each trial. This is important, since clearly if one input is 0, that input will not affect the value of the node, and so we don't want to include the error from that input in the calculation of the weight adjustment. Multiplying the error by the input matrix allows us to disregard the errors found when the input is 0, as seen by the calculation of `adjustments[0]`. The second, fourth and sixth trial errors are not important to the adjustment of the first node weight, since they are 0, so they didn't impact the node value calculation, which is reflected in the calculation, since the first node weight adjustment is only the sum of the first, third and fifth trial errors.

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
Clearly the pattern returns `x+1`, given some `x`. Notice we are using the ReLU activation function. This function takes in some `x` and simply returns the maximum of `x` and 0. Since we want our output to be an unbouded integer, we are restricted from using activation functions like sigmoid or tanh, which is why we choose ReLU. The output is as follows:
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
