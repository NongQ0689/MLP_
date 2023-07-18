import numpy as np

class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Initialize weights randomly
        self.weights1 = np.random.randn(self.input_dim, self.hidden_dim)
        self.weights2 = np.random.randn(self.hidden_dim, self.output_dim)
        
        # Learning rate
        self.learning_rate = 0.01

    def forward_pass(self, X):
        # Forward pass through the network
        self.hidden_layer = np.dot(X, self.weights1)
        self.hidden_activation = self.sigmoid(self.hidden_layer)
        self.output_layer = np.dot(self.hidden_activation, self.weights2)
        self.output_activation = self.sigmoid(self.output_layer)
        return self.output_activation

    def backward_pass(self, X, y, y_pred):
        # Backward pass through the network
        error = y - y_pred
        
        # Compute gradients
        delta_output = error * self.sigmoid_derivative(self.output_layer)
        delta_hidden = np.dot(delta_output, self.weights2.T) * self.sigmoid_derivative(self.hidden_layer)
        
        # Update weights
        self.weights2 += self.learning_rate * np.dot(self.hidden_activation.T, delta_output)
        self.weights1 += self.learning_rate * np.dot(X.T, delta_hidden)
        
    def train(self, X, y, epochs):
        for epoch in range(epochs):
            # Forward pass
            output = self.forward_pass(X)
            
            # Backward pass
            self.backward_pass(X, y, output)
            
            # Print loss every 100 epochs
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def sigmoid(self, x):
        x_clipped = np.clip(x, -500, 500)
        return  1 / (1 + np.exp(-x_clipped))
    
    def sigmoid_derivative(self, x):
        return x * (1 - np.clip(x, -500, 500))
    
# Example data
#data = np.array([[10, 2, 30],
                 #[5, 6, 90],
                 #[3, 8, 120]])

def read_data_from_file(filename):
    data = []
    with open(filename, "r") as file:
        for line in file:
            values = [int(x) for x in line.split()]
            data.append(values)
    return data

def normalize_data(data):
    normalized_list = []
    for sublist in data:
        min_val = min(sublist)
        max_val = max(sublist)
        normalized_sublist = [(x - min_val) / (max_val - min_val) for x in sublist]
        normalized_list.append(normalized_sublist)
    return normalized_list

def denormalize_data(data, original_data):
    denormalized_list = []
    for i, sublist in enumerate(data):
        min_val = min(original_data[i])
        max_val = max(original_data[i])
        denormalized_sublist = [(x * (max_val - min_val) + min_val) for x in sublist]
        denormalized_list.append(denormalized_sublist)
    return denormalized_list



data    = read_data_from_file("Data.txt")
data_p  = read_data_from_file("Data_p.txt")


normalized_dataX = np.array(normalize_data(data))

# Split the data into input (X) and output (y) arrays
X = normalized_dataX[:, :8]
y = normalized_dataX[:, 8:]

print(X)

# Create a neural network with 2 input units, 2 hidden units, and 1 output unit
#nn = NeuralNetwork(8, 10, 1)

# Train the neural network
#nn.train(X, y, epochs=10000)

# Test the neural network
#output = nn.forward_pass(X)
#print("Predicted Output:")
#print(output)
#print(output * (max_valsP[2] - min_valsP[2]) + min_valsP[2])  # Denormalize the output
