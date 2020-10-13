# MiniNeuralNets
A framework for mini neural networks (single node networks, also known as perceptrons), written from scratch in python

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
Notice the output is simply the last element in the input, it's difficult for a perceptron to find more complex patterns, after all, it's only one node. Now you have a trained perceptron, you can make predictions with it using `nn.predict(x)`, for some input `x`. The method returns the pre-activation output, the prediction (rounded activation function on the output), as well as the confidence of the choice repsectively.
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
As you can see, we've successfully trained our perceptron to model the pattern given, and it was able to predict the result of an input it's never seen before, with a confidence of 99.4%.

## Backpropagation.py
Backpropagation is a method of adjusting the weights in a neural network. The idea behind this algorithm is to significantly adjust the weights when the error is large, and make smaller adjustments the weights when the error is small. This can be done by setting the adjustment values to be the matrix of inputs multiplied by the vector of errors. The vector of errors is calculated element-wise by multiplying the difference between the output and the actual answer by the derivative of the activation function of our choosing. The result is the adjustment vector, where the ith element is the amount that the ith node's weight should change, and so we complete the backpropagation by adding the adjustment vector to the weights, which leaves us with slightly more accurate weights. Notice that when an input is 0, the value that input contributes to the node will always be 0, and so we've learned nothing about the weight of that connection. This is why we multiply the matrix of inputs by the vectors of errors, so that in the cases where the input is 0, the weight adjustment for that input will also be 0, preventing us from adjusting a weight we know nothing about.
