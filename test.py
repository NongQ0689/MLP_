import numpy as np

def __init__(self, input_size, hidden_size1, hidden_size2, output_size, learning_rate, momentum_rate):
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate

        # Initialize weights with random values
        self.weights_input_hidden1 = np.random.randn(self.hidden_size1, self.input_size)
        self.weights_hidden1_hidden2 = np.random.randn(self.hidden_size2, self.hidden_size1)
        self.weights_hidden2_output = np.random.randn(self.output_size, self.hidden_size2)

        # Initialize biases with random values
        self.bias_hidden1 = np.random.randn(self.hidden_size1, 1)
        self.bias_hidden2 = np.random.randn(self.hidden_size2, 1)
        self.bias_output = np.random.randn(self.output_size, 1)

        # Initialize momentum values to zero
        self.velocity_weights_input_hidden1 = np.zeros_like(self.weights_input_hidden1)
        self.velocity_weights_hidden1_hidden2 = np.zeros_like(self.weights_hidden1_hidden2)
        self.velocity_weights_hidden2_output = np.zeros_like(self.weights_hidden2_output)

        self.velocity_bias_hidden1 = np.zeros_like(self.bias_hidden1)
        self.velocity_bias_hidden2 = np.zeros_like(self.bias_hidden2)
        self.velocity_bias_output = np.zeros_like(self.bias_output)

    def forward(self, input_data):
        # ... (unchanged)

    def backward(self, input_data, target):
        # ... (unchanged)

        # Update weights and biases with momentum
        self.velocity_weights_hidden2_output = self.momentum_rate * self.velocity_weights_hidden2_output + \
            self.learning_rate * np.dot(output_delta, hidden2_outputs.T)
        self.velocity_bias_output = self.momentum_rate * self.velocity_bias_output + \
            self.learning_rate * output_delta

        self.velocity_weights_hidden1_hidden2 = self.momentum_rate * self.velocity_weights_hidden1_hidden2 + \
            self.learning_rate * np.dot(hidden2_delta, hidden1_outputs.T)
        self.velocity_bias_hidden2 = self.momentum_rate * self.velocity_bias_hidden2 + \
            self.learning_rate * hidden2_delta

        self.velocity_weights_input_hidden1 = self.momentum_rate * self.velocity_weights_input_hidden1 + \
            self.learning_rate * np.dot(hidden1_delta, input_data.T)
        self.velocity_bias_hidden1 = self.momentum_rate * self.velocity_bias_hidden1 + \
            self.learning_rate * hidden1_delta

        # Apply updates with momentum
        self.weights_hidden2_output -= self.velocity_weights_hidden2_output
        self.bias_output -= self.velocity_bias_output

        self.weights_hidden1_hidden2 -= self.velocity_weights_hidden1_hidden2
        self.bias_hidden2 -= self.velocity_bias_hidden2

        self.weights_input_hidden1 -= self.velocity_weights_input_hidden1
        self.bias_hidden1 -= self.velocity_bias_hidden1