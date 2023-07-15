import numpy as np

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Define the Multilayer Perceptron class
class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize the weights randomly
        self.weights1 = np.random.uniform(size=(input_size, hidden_size))
        self.weights2 = np.random.uniform(size=(hidden_size, output_size))

    def forward(self, X):
        # Perform the forward pass through the network
        self.hidden_layer = sigmoid(np.dot(X, self.weights1))
        self.output_layer = sigmoid(np.dot(self.hidden_layer, self.weights2))
        return self.output_layer

    def backward(self, X, y, learning_rate):
        # Perform the backward pass and update the weights
        output_error = y - self.output_layer
        print(self.output_layer)
        output_delta = output_error * sigmoid_derivative(self.output_layer)

        hidden_error = np.dot(output_delta, self.weights2.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_layer)

        self.weights2 += learning_rate * np.dot(self.hidden_layer.T, output_delta)
        self.weights1 += learning_rate * np.dot(X.T, hidden_delta)

    def train(self, X, y, epochs, learning_rate):
        # Train the network for the specified number of epochs
        i=0
        for epoch in range(epochs):
            i+=1

            self.forward(X)
            self.backward(X, y, learning_rate)

            if i % 10000 == 0 : 
                print(i)

    def predict(self, X):
        # Make predictions using the trained network {{{{{ self.forward(X)  }}}}}
        output = self.forward(X)
        return output

def output_dec(output) :
    sum = 0
    sum_bin = output_bin(output)
    for i in range(len(sum_bin)):
        sum += (2**(9-i))*sum_bin[i]
    return sum

def output_bin(output) :
    for i in range(len(output)):
        output[i] = 1 if output[i] > 0.5 else 0
    return output

# Example usage
X = np.array([
    [ 0,1,0,1,0,1,1,1,1,0,0,1,0,1,1,0,0,0,0,0,0,1,0,1,1,0,0,1,0,0,0,1,0,1,1,0,1,0,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0 ] ,
[ 0,1,0,1,1,0,0,0,0,0,0,1,0,1,1,0,0,1,0,0,0,1,0,1,1,0,1,0,0,0,0,1,0,1,1,0,1,1,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,1,1,0 ] ,
[ 0,1,0,1,1,0,0,1,0,0,0,1,0,1,1,0,1,0,0,0,0,1,0,1,1,0,1,1,0,0,0,1,0,1,1,0,1,1,0,1,0,1,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,1,1,0,0,1,0,0,1,0,1,1,1,0 ] ,
[ 0,1,0,1,1,0,1,0,0,0,0,1,0,1,1,0,1,1,0,0,0,1,0,1,1,0,1,1,0,1,0,1,0,1,1,0,1,1,1,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,1,1,0,0,1,0,0,1,0,1,1,1,0,0,1,0,0,1,1,0,1,0,0 ] ,
[ 0,1,0,1,1,0,1,1,0,0,0,1,0,1,1,0,1,1,0,1,0,1,0,1,1,0,1,1,1,0,0,1,0,1,1,1,0,0,0,0,0,1,0,0,1,0,0,1,1,0,0,1,0,0,1,0,1,1,1,0,0,1,0,0,1,1,0,1,0,0,0,1,0,0,1,1,1,0,1,1 ] ,
[ 0,1,0,1,1,0,1,1,0,1,0,1,0,1,1,0,1,1,1,0,0,1,0,1,1,1,0,0,0,0,0,1,0,1,1,1,0,0,0,1,0,1,0,0,1,0,1,1,1,0,0,1,0,0,1,1,0,1,0,0,0,1,0,0,1,1,1,0,1,1,0,1,0,1,0,0,0,0,1,0 ] ,
[ 0,1,0,1,1,0,1,1,1,0,0,1,0,1,1,1,0,0,0,0,0,1,0,1,1,1,0,0,0,1,0,1,0,1,1,1,0,0,0,0,0,1,0,0,1,1,0,1,0,0,0,1,0,0,1,1,1,0,1,1,0,1,0,1,0,0,0,0,1,0,0,1,0,1,0,0,0,1,1,1 ] ,
[ 0,1,0,1,1,1,0,0,0,0,0,1,0,1,1,1,0,0,0,1,0,1,0,1,1,1,0,0,0,0,0,1,0,1,1,1,0,0,0,0,0,1,0,0,1,1,1,0,1,1,0,1,0,1,0,0,0,0,1,0,0,1,0,1,0,0,0,1,1,1,0,1,0,1,0,0,1,0,1,0 ] ,
[ 0,1,0,1,1,1,0,0,0,1,0,1,0,1,1,1,0,0,0,0,0,1,0,1,1,1,0,0,0,0,0,1,0,1,1,0,1,1,1,1,0,1,0,1,0,0,0,0,1,0,0,1,0,1,0,0,0,1,1,1,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,1,0,1 ] ,
[ 0,1,0,1,1,1,0,0,0,0,0,1,0,1,1,1,0,0,0,0,0,1,0,1,1,0,1,1,1,1,0,1,0,1,1,0,1,1,0,1,0,1,0,1,0,0,0,1,1,1,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,1,0,1,0,1,0,1,0,0,1,1,0,1 ] ,
[ 0,1,0,1,1,1,0,0,0,0,0,1,0,1,1,0,1,1,1,1,0,1,0,1,1,0,1,1,0,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,1,0,1,0,1,0,1,0,0,1,1,0,1,0,1,0,1,0,0,1,0,1,1 ] ,
[ 0,1,0,1,1,0,1,1,1,1,0,1,0,1,1,0,1,1,0,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,0,0,0,1,0,1,0,0,1,1,0,1,0,1,0,1,0,0,1,1,0,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,0,0 ] ,
[ 0,1,0,1,1,0,1,1,0,1,0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,0,0,0,1,0,1,1,0,0,1,0,1,0,1,0,1,0,0,1,1,0,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,0,0,0,1,0,1,0,0,1,0,0,0 ] ,
[ 0,1,0,1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,0,0,0,1,0,1,1,0,0,1,0,1,0,1,0,1,1,0,0,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,0,0,1,0,0,0,0,1,0,1,0,0,1,0,0,0,0,1,0,1,0,0,0,1,0,1 ] ,
[ 0,1,0,1,1,0,1,0,0,0,0,1,0,1,1,0,0,1,0,1,0,1,0,1,1,0,0,0,1,1,0,1,0,1,1,0,0,0,1,0,0,1,0,1,0,0,1,0,0,0,0,1,0,1,0,0,1,0,0,0,0,1,0,1,0,0,0,1,0,1,0,1,0,1,0,0,0,0,1,0 ] 
])

y = np.array([
    [0,1,0,1,0,0,1,0,1,0],
[0,1,0,1,0,0,1,1,0,1],
[0,1,0,1,0,0,1,1,0,1],
[0,1,0,1,0,0,1,0,1,1],
[0,1,0,1,0,0,1,0,0,0],
[0,1,0,1,0,0,1,0,0,0],
[0,1,0,1,0,0,0,1,0,1],
[0,1,0,1,0,0,0,0,1,0],
[0,1,0,1,0,0,0,0,0,0],
[0,1,0,0,1,1,1,1,0,1],
[0,1,0,0,1,1,1,0,1,1],
[0,1,0,0,1,1,0,1,1,0],
[0,1,0,0,1,1,0,0,1,1],
[0,1,0,0,1,1,0,0,0,1],
[0,1,0,0,1,1,0,0,0,1]
])

# Create an MLP with 3 input neurons, 4 hidden neurons, and 1 output neuron
mlp = MLP(80, 10, 10)

# Train the MLP for 10000 epochs with a learning rate of 0.1
mlp.train(X, y, epochs=10000, learning_rate=0.1)

# Make predictions on new data
test_data = np.array(
    [ 0,1,0,1,1,1,0,0,0,1,0,1,0,1,1,1,0,0,0,0,0,1,0,1,1,1,0,0,0,0,0,1,0,1,1,0,1,1,1,1,0,1,0,1,0,0,0,0,1,0,0,1,0,1,0,0,0,1,1,1,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,1,0,1 ]
    )
predictions = mlp.predict(test_data)
print(predictions)
print(output_bin(predictions))
print(output_dec(predictions))