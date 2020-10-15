# MiniNeuralNets
A framework for mini neural networks (single node networks, also known as perceptrons), written from scratch in python. The goal of the project is to demystify the workings of a neural network and various training algorithms by providing code written from scratch of the simplest neural network one could have. You can easily train a mini neural network to try to model a linear pattern of your choosing, and swap between training algorithms and cost functions to get a sense of what works, and why.

## main.py
This script acts as the controller for the mini neural networks. You're able to instantiate a perceptron with
```python
nn = Perceptron(activation=sigmoid, error=mse)
```
This will store a perceptron in nn, which contains the weights, activation function, error function and accuracy for the model. To train the model, you call `nn.train`, passing the training algorithm of your choice, along with the arguments needed for your chosen training algorithm. As an example, to train a model using gradient descent on inputs `x`, answers `y`, with 1000 iterations, you pass the following parameters:
```python
nn.train(gradient_descent, x, y, 1000)
```
Then, to make a prediction of some input `input` using the trained model, you call
```python
output, prediction, confidence = nn.predict(input)
```
where `output` is the actual output of the model, `prediction` is the rounded `output`, and `confidence` is the confidence the model has for it's prediction.

## GradientDescent.py
The gradient descent algorithm is a method of adjusting the weights of the model in an attempt to minimize the error function, and does so using the gradient of the error function, which is calculated by backpropagation. Backpropagation and gradient descent are often used interchangably, however, gradient descent is the actual training algorithm, while backpropagation is a generalization of the computation of the gradient. The idea of the algorithm is to adjust the weights significantly when the error is large, and to make progressively smaller adjustments the closer the error is to 0. The 'gradient' in 'gradient descent' comes from the algorithms use of the gradient of the error function to determine the direction of the adjustment. For each weight, we need to calculate the delta value, which is found by multiplying the derivative of the activation function of the neuron value by the derivative of the error function. Then, to calculate the adjustment amount for that weight, we multiply the delta value by the input value attached to the weight, and we are left with the amount we need to subtract to the weight in order to slightly improve the accuracy of the model.

We will now train a model using gradient descent and the sigmoid activation function to predict the output of the following pattern, where each row in `x` is a trial, and the desired output of the trial is stored in the corresponding index of `y`
```python 
x = np.array([[1, 1, 0, 0],
              [0, 0, 1, 0],
              [1, 0, 0, 1],
              [0, 1, 1, 1]])

y = np.array([0, 0, 1, 1])

nn = Perceptron(activation=sigmoid, error=mse)
# x are the training inputs, y are training labels, 10000 iterations and a learning rate of 2
nn.train(gradient_descent, x, y, 10000, 2)

input_vals = np.array([0,1,1,0])
nn.predict(input_vals)
```
Notice the output is simply the last element in the input. It's difficult for a perceptron to find more complex patterns, after all, it's only one node. The output is as follows:
```python
# OUTPUT
INPUT 
 [0 1 1 0]
PREDICTION 	 CONFIDENCE 
 0.0 		 0.9946591601887982
WEIGHTS 
 [-1.07974768 -0.14963171 -1.10147694 10.43115913 -3.97590856]
ERROR 
 2.9821391002328183e-05
```
So, we have trained a perceptron to model the given pattern, and it has produced the correct result with a confidence of 99.47%, on an input it has never seen before. Notice that each weight corresponds to it's respective input, i.e., the first value in the input pattern has a weight of `-1.0797`, the last value in the input pattern has a weight of `10.431`, and the bias has a weight of `-3.976`. From these weights, it's easy to see how the model calculates the result given an input; it essentially ignores the first 3 inputs, making the value of the neuron entirely dependent on the last input value, which is exactly how it is expected to model the pattern.

## GeneticAlgorithm.py
The genetic algorithm tries to mimic evolution, by adjusting the weights of a given model until it's found optimal weights. It's typically used in a unsupervised manner, measuring the performance of the model based on the 'score' it obtains in an environment it's being trained in. However, since this is a learning exercise, we slightly modify the algorithm to be supervised, by calculating the score of the model based on how close the output is to the actual answer. There are 5 main steps to this algorithm:

1. **Generate Agents:**
We randomly initialize `num_agents` agents in the initial generation. An agent is one set of weights for the model.
2. **Measure Fitness:**
We find some way to calculate the fitness of each agent. In our case, since we have the desired outputs, we use the error of the model's prediction (using a user-passed error function) to rank the fitnesses of the agents. The fitness is a measure of how accurate the model fits the training inputs.
3. **Selection:**
We choose some number of agents to reproduce, where the higher the agent's fitness, the more likely they are to be selected for reproduction.
4. **Reproduction:**
Out of the pool of selected agents, we randomly choose two to reproduce, until we have close to `num_agents` agents (roughly 75%, chosen fairly arbitrarily). The reproduction is the creation of a child, where each weight in the child is chosen to be one of the two parent's weight in that position. We also include a 10% chance for any given weight to be set to 0. This random mutation allows for new patterns to emerge (in case the model gets stuck in a local minimum), and it also simulates the idea of dropout, which again prevents the model from being stuck in a local minimum, and allows it to explore more possibilities.
5. **Create New Generation:**
When we have a number of children that are close to the number of agents we started with (75%, see step 4), we fill the rest of the generation with randomized weights, again to allow new patterns to emerge.

This process is performed `generations` times, and the assumption of the algorithm is that with each new generation, the average fitness of the agents in the generation will have slightly improved, because the stronger performing weights were more likely to reproduce, passing down their DNA. We will now give an example of the genetic algorithm in action, using it to train a perceptron to model a very simple pattern, where each row in `x` is a trial, and the desired output of the trial is stored in the corresponding index of `y`

```python
x = np.array([[0],
              [1],
              [2]])
y = np.array([1, 2, 3])

nn = Perceptron(activation=relu, error=mae)
# we will simulate 1000 generations, with each having 100 agents, and return the model early if the error is <= 0.002
nn.train(genetic_algorithm, x, y, 1000, 100, 0.002)

input_vals = np.array([100])
nn.predict(input_vals)
```
Clearly the pattern returns `x+1`, given some `x`, and notice we have chosen to use the `ReLU` activation function. This function takes in some `x` and simply returns the maximum of `x` and `0`. Since we want our output to be an unbouded integer, we are restricted from using activation functions such as `sigmoid` or `tanh`, which is why we choose `ReLU`. Also notice that since `ReLU` returns `max(0,x)`, we aren't able to train the model to predict negative numbers. The output is as follows:
```python
# OUTPUT
INPUT 
 [100]
PREDICTION 	 CONFIDENCE 
 101.0 		 0.9762138654645582
WEIGHTS 
 [0.99977005 0.99920929]
ERROR 
 0.001020665568131296
```
So, we've successfully trained the perceptron to model the function `x+1` (for relatively small inputs). Consider the weights, noticing that the first weight is the connection between the input and the node value (which is the output, since this is a perceptron), and the second weight is the connection between the bias (value=1) and the node value. It isn't hard to see that if the model was 100% accurate, the weights would be `[1, 1]`, since to calculate the node value, you add the result of the first weight multiplied by the input and the second weight mulitplied by the bias. When both weights are `1`, this simplifies to `node_value = ReLU(input + 1) = input + 1`, which is exactly our desired relationship.
