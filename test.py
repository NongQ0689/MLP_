import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights with random values
        self.weights_input_hidden = np.random.randn(self.hidden_size, self.input_size)
        self.weights_hidden_output = np.random.randn(self.output_size, self.hidden_size)

        # Initialize biases with random values
        self.bias_hidden = np.random.randn(self.hidden_size, 1)
        self.bias_output = np.random.randn(self.output_size, 1)

    def forward(self, input_data):
        # Convert input data to column vector
        input_data = np.array(input_data).reshape(self.input_size, 1)

        # Calculate hidden layer activations
        hidden_activations = np.dot(self.weights_input_hidden, input_data) + self.bias_hidden
        hidden_outputs = self.sigmoid(hidden_activations)

        # Calculate output layer activations
        output_activations = np.dot(self.weights_hidden_output, hidden_outputs) + self.bias_output
        output = self.sigmoid(output_activations)

        return output

    def backward(self, input_data, target):
        # Convert input data and target to column vectors
        input_data = np.array(input_data).reshape(self.input_size, 1)
        target = np.array(target).reshape(self.output_size, 1)

        # Perform forward pass
        hidden_activations = np.dot(self.weights_input_hidden, input_data) + self.bias_hidden
        hidden_outputs = self.sigmoid(hidden_activations)
        output_activations = np.dot(self.weights_hidden_output, hidden_outputs) + self.bias_output
        output = self.sigmoid(output_activations)

        # Calculate output layer error
        output_error = output - target
        output_delta = output_error * self.sigmoid_derivative(output_activations)

        # Calculate hidden layer error
        hidden_error = np.dot(self.weights_hidden_output.T, output_delta)
        hidden_delta = hidden_error * self.sigmoid_derivative(hidden_activations)

        # Update weights and biases
        self.weights_hidden_output -= self.learning_rate * np.dot(output_delta, hidden_outputs.T)
        self.bias_output -= self.learning_rate * output_delta
        self.weights_input_hidden -= self.learning_rate * np.dot(hidden_delta, input_data.T)
        self.bias_hidden -= self.learning_rate * hidden_delta

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            for input_data, target in zip(X, y):
                # Perform backpropagation
                self.backward(input_data, target)

            # Print loss every 100 epochs
            if epoch % 100 == 0:
                output = np.array([self.forward(x)[0] for x in X])
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))




