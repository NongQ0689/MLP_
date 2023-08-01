import numpy as np
import matplotlib.pyplot as plt

##################################################  Data

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

##################################################  MLP

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
def sigmoid_derivative(x):
    return x * (1 - x)

def init_w0(input_size, hidden_size, output_size):
    global weights_input_hidden, weights_hidden_output, weights_bias_hidden, weights_bias_output
    
    weights_input_hidden = np.random.randn(hidden_size, input_size)
    weights_hidden_output = np.random.randn(output_size, hidden_size)
    weights_bias_hidden = np.random.randn(hidden_size, 1)
    weights_bias_output = np.random.randn(output_size, 1)

    global velocity_weights_input_hidden, velocity_weights_hidden_output, velocity_weights_bias_hidden, velocity_weights_bias_output
    velocity_weights_input_hidden = np.zeros_like(weights_input_hidden)
    velocity_weights_hidden_output = np.zeros_like(weights_hidden_output)
    velocity_weights_bias_hidden = np.zeros_like(weights_bias_hidden)
    velocity_weights_bias_output = np.zeros_like(weights_bias_output)

def forward(input_data):
    input_data = np.array(input_data)

    hidden_activations = np.dot(weights_input_hidden, input_data.T) + weights_bias_hidden
    hidden = sigmoid(hidden_activations)

    output_activations = np.dot(weights_hidden_output, hidden) + weights_bias_output
    output = sigmoid(output_activations)

    return output

def train(input_data , output_data , N , target_mse , lr, momentum_rate ):
    global weights_input_hidden, weights_hidden_output, weights_bias_hidden, weights_bias_output
    global velocity_weights_input_hidden, velocity_weights_hidden_output, velocity_weights_bias_hidden, velocity_weights_bias_output
    
    epochs = 0
    mse = 1
    mse_history = []  # List to store the MSE for each epoch

    while  epochs < N and mse > target_mse :
        input_data = np.array(input_data)

        hidden_activations = np.dot(weights_input_hidden, input_data.T) + weights_bias_hidden
        hidden = sigmoid(hidden_activations)

        output_activations = np.dot(weights_hidden_output, hidden) + weights_bias_output
        output = sigmoid(output_activations)

        output_error = output_data - output
        mse = np.mean(output_error**2) # Calculate the mean squared error
        
        output_delta = output_error * sigmoid_derivative(output)
        
        # Update output layer weight and bias
        velocity_weights_hidden_output = (momentum_rate * velocity_weights_hidden_output) + (lr * np.dot(output_delta, hidden.T) / len(input_data))
        weights_hidden_output += velocity_weights_hidden_output
        velocity_weights_bias_output = (momentum_rate * velocity_weights_bias_output) + (lr * np.mean(output_delta, axis=1, keepdims=True))
        weights_bias_output += velocity_weights_bias_output

        # Update hidden layer weight and bias
        hidden_error = np.dot(weights_hidden_output.T, output_delta)
        hidden_delta = hidden_error * sigmoid_derivative(hidden)
        velocity_weights_input_hidden = (momentum_rate * velocity_weights_input_hidden) + (lr * np.dot(hidden_delta, input_data) / len(input_data))
        weights_input_hidden += velocity_weights_input_hidden
        velocity_weights_bias_hidden = (momentum_rate * velocity_weights_bias_hidden) + (lr * np.mean(hidden_delta, axis=1, keepdims=True))
        weights_bias_hidden += velocity_weights_bias_hidden

        epochs += 1
        if (epochs) % 5000 == 0:
            print(f"Epoch: {epochs}, MSE: {mse}")

        mse_history.append(mse)  # Store the current MSE in the list

    return mse_history

##################################################  MAPE  MAE:

def mean_absolute_percentage_error(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100

def mean_absolute_error(actual, predicted):
    return np.mean(np.abs(actual - predicted))

##################################################

data = read_data_from_file("shuffled_data.txt")
minimum, maximum = min_max(data)
normalized_data = normalize_data(data, minimum, maximum)
X, Y  = extract_features_labels(normalized_data)

init_w0(8,16,1)
mse_history = train(X, Y, N = 100000 , target_mse = 0.000001, lr = 0.7, momentum_rate = 0.9 )

data_p = read_data_from_file("Data_P.txt")
normalized_data_p = normalize_data(data_p, minimum, maximum)   
Z, Z2 = extract_features_labels(normalized_data_p)
prediction = forward(Z)

denormalize_predictions = denormalize_data(prediction, minimum, maximum)
print("Predictions:", denormalize_predictions)

mape = mean_absolute_percentage_error(Z2, prediction)
mae = mean_absolute_error(Z2, prediction)
accuracy = 100 - mape

print(f"Accuracy: {accuracy:.2f}%")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Mean Absolute Error (MAE): {mae:.2f}")

################################################### Plotting the graph

plt.plot(range(1, len(mse_history) + 1), mse_history, marker='o')
plt.xscale('log')  
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Training Progress')
plt.grid(True)
plt.show()