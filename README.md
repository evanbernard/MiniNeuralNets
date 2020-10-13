# MiniNeuralNets
A collection of mini neural networks written from scratch in python


## Backpropagation.py
This neural network uses backpropagation to adjust the weights of the perceptron. The perceptron has the standard structure, with any number of inputs, and only one neuron, acting as the output. The idea behind backpropagation is to significantly adjust the weights when the error is large, and adjust the weights less when the error is small. This can be done by setting the adjustment to be the matrix of inputs multiplied by the vector of errors. The vector of errors is calculated element-wise by multiplying the difference between the output and the actual answer by the derivative of the sigmoid function (our activation function). The result is the adjustment vector, where the ith element is the amount that the ith node's weight should change, and so we complete the backpropegation by adding the adjustment vector to the weights, and doing it over again, now with slightly more accurate weights.
```python
training_inputs = np.array([[1, 1, 0, 0],
                            [0, 0, 1, 0],
                            [1, 0, 1, 0],
                            [0, 0, 0, 1],
                            [1, 0, 1, 1],
                            [0, 1, 1, 1]])
```
Each row in `training_inputs` is a question, where the answer for each row is obtained by taking the last element in the list.
```python
# answers to the input patterns
training_outputs = np.array([0, 0, 0, 1, 1, 1])
```
After training the model, we want to test it on a pattern it has not seen before, so we will use the following pattern
```python
input_vals = [0, 1, 1, 0]
```
Now we can train the model on `training_inputs` and have it predict the new pattern:
```python
weights = train(training_inputs, training_outputs, num_iterations=10000)
input_vals = np.array([0, 1, 1, 0])
_, prediction, error = predict(weights, input_vals)
```
The resulting output is as follows:
```python
ACCURACY: 
 0.9999577269693771
INPUT 
 [0, 1, 1, 0]
PREDICTION 		 ERROR 
 0.0 			 2.731818946727216e-05
```
So, using backpropagation, we've adjusted the weights of a single-neuron neural network in order to accurately predict the answers to the simple pattern question.
