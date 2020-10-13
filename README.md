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
Backpropagation is a method of adjusting the weights in a neural network. The central idea behind this algorithm is to significantly adjust the weights when the error is large, and make smaller adjustments the weights when the error is small, where the direction of the adjustment is relative to the gradient of the activation function. This can be done by setting the weight adjustment values to be the vector of errors multiplied by the matrix of inputs. The vector of errors is calculated element-wise by multiplying the difference between the output and the actual answer by the derivative of the activation function of our choosing. The result is the adjustment vector, where the ith element is the amount that the ith node's weight should change, and so we complete the backpropagation algorithm by adding the adjustment vector to the weights, leaving us with slightly more accurate weights. The calculation of the weight adjustment vector can be seen here:
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
Recall that the first column in the inputs matrix would be the first nodes in the trials (one row of inputs is one trial), and `ei` is the error for the ith input. So, `adjustments[0]` is the dot product of the errors from each trial multiplied by the first node input of each trial. This is important, since clearly if one input is 0, that input will not affect the value of the node, and so we don't want to include the error from that input in the calculation of the weight adjustment. Multiplying the error by the input matrix allows us to disregard the errors found when the input is 0, as seen by the calculation of `adjustments[0]`, the second, fourth and sixth trial errors are not important to the adjustment of the first node weight, since they are 0, so they didn't impact the node value calculation. 
