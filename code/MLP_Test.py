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
        output_delta = output_error * sigmoid_derivative(self.output_layer)

        hidden_error = np.dot(output_delta, self.weights2.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_layer)

        self.weights2 += learning_rate * np.dot(self.hidden_layer.T, output_delta)
        self.weights1 += learning_rate * np.dot(X.T, hidden_delta)

    def train(self, X, y, epochs, learning_rate):
        # Train the network for the specified number of epochs
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, learning_rate)

    def predict(self, X):
        # Make predictions using the trained network
        return self.forward(X)


# Example usage
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# Create an MLP with 3 input neurons, 4 hidden neurons, and 1 output neuron
mlp = MLP(3, 4, 1)

# Train the MLP for 10000 epochs with a learning rate of 0.1
mlp.train(X, y, epochs=10000, learning_rate=0.1)

# Make predictions on new data
test_data = np.array([[0, 0, 1],
                      [0, 1, 1]])
predictions = mlp.predict(test_data)
print(predictions)
