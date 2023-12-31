import numpy as np
import matplotlib.pyplot as plt

############################################################################


def read_data_from_file(filename):
    data = []
    with open(filename, "r") as file:
        for line in file:
            values = [int(x) for x in line.split()]
            data.append(values)
    return data

def min_max(data):
    flat_data = [item for sublist in data for item in sublist]
    minimum = min(flat_data)
    maximum = max(flat_data)
    return minimum, maximum

def normalize_data(data, min_value, max_value):

    # Apply min-max normalization to each value
    normalized_data = [
        [(value - min_value) / (max_value - min_value) for value in sublist]
        for sublist in data
    ]
    return normalized_data


def denormalize_data(normalized_data, min_value, max_value):
    # Apply denormalization to each value
    denormalized_data = [
        [(value * (max_value - min_value)) + min_value for value in sublist]
        for sublist in normalized_data
    ]
    return denormalized_data

def denormalize_data(value, min_value, max_value): 
    denormalize_data = (value * (max_value - min_value)) + min_value
    return denormalize_data


def extract_features_labels(data):
    X = [sublist[:8] for sublist in data]
    Y = [sublist[-1] for sublist in data]
    return X, Y

############################################################################

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, momentum_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate

        # Initialize weights with random values
        self.weights_input_hidden = np.random.randn(self.hidden_size, self.input_size)
        self.weights_hidden_output = np.random.randn(self.output_size, self.hidden_size)

        # Initialize biases with random values
        self.bias_hidden = np.random.randn(self.hidden_size, 1)
        self.bias_output = np.random.randn(self.output_size, 1)

        # Initialize velocities with zeros
        self.velocity_weights_input_hidden = np.zeros((self.hidden_size, self.input_size))
        self.velocity_weights_hidden_output = np.zeros((self.output_size, self.hidden_size))
        self.velocity_bias_hidden = np.zeros((self.hidden_size, 1))
        self.velocity_bias_output = np.zeros((self.output_size, 1))

    def forward(self, input_data):
        # Convert input data to a 2D array
        input_data = np.array(input_data)
        
        # Perform the forward pass for the batch of input samples
        hidden_activations = np.dot(self.weights_input_hidden, input_data.T) + self.bias_hidden
        hidden_outputs = self.sigmoid(hidden_activations)
        
        output_activations = np.dot(self.weights_hidden_output, hidden_outputs) + self.bias_output
        output = self.sigmoid(output_activations)
        
        return output.T  # Transpose the output to match the input_data shape

    # Rest of the class implementation...

    
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

        # Update weights and biases with momentum
        self.velocity_weights_hidden_output = (self.momentum_rate * self.velocity_weights_hidden_output) + \
            (self.learning_rate * np.dot(output_delta, hidden_outputs.T))
        self.velocity_bias_output = (self.momentum_rate * self.velocity_bias_output) + \
            (self.learning_rate * output_delta)

        self.velocity_weights_input_hidden = (self.momentum_rate * self.velocity_weights_input_hidden) + \
            (self.learning_rate * np.dot(hidden_delta, input_data.T))
        self.velocity_bias_hidden = (self.momentum_rate * self.velocity_bias_hidden) + \
            (self.learning_rate * hidden_delta)

        # Apply updates with momentum
        self.weights_hidden_output -= self.velocity_weights_hidden_output
        self.bias_output -= self.velocity_bias_output

        self.weights_input_hidden -= self.velocity_weights_input_hidden
        self.bias_hidden -= self.velocity_bias_hidden

    def train(self, X, y, epochs):
        losses = []  # To store loss values for each epoch
        for epoch in range(epochs):
            for input_data, target in zip(X, y):
                # Perform backpropagation
                self.backward(input_data, target)

            # Calculate loss for the current epoch
            output = np.array([self.forward(x)[0] for x in X])
            loss = np.mean(np.square(y - output))
            losses.append(loss)

            # Print loss every 100 epochs
            if (epoch + 1) % 500 == 0:
                print(f"Epoch {epoch + 1}, Loss: {loss}")

        # Plot the graph
        plt.plot(range(1, epochs + 1), losses)  # Use range(1, epochs + 1) as x-axis for epochs
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss vs. Epochs')
        plt.show()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))


############################################################################

data = read_data_from_file("shuffled_data.txt")

minimum, maximum = min_max(data)
normalized_data = normalize_data(data, minimum, maximum)

X, Y  = extract_features_labels(normalized_data)
nn = NeuralNetwork(8, 8, 1, 0.000001, 0.1)
nn.train(X, Y, 500)


data_p = read_data_from_file("Data_P.txt")
normalized_data_p = normalize_data(data_p, minimum, maximum)   #min max คนละตัววว
Z, Z2 = extract_features_labels(normalized_data_p)


prediction = nn.forward(Z)

for i, P in enumerate(prediction):
    print(f"Prediction for Data_P {i+1}: {denormalize_data( P , minimum , maximum )}")


