import numpy as np
import matplotlib.pyplot as plt1

def read_data_from_file(filename):
    x = []
    y = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        for i in range(1, len(lines), 3):
            line = lines[i].strip()
            numbers = [float(x) for x in line.split()]
            x.append(numbers)

        for i in range(2, len(lines), 3):
            line = lines[i].strip()
            numbers = [float(x) for x in line.split()]
            y.append(numbers)

    return x , y


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

    return weights_input_hidden, weights_hidden_output, weights_bias_hidden, weights_bias_output

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
        output_data = np.array(output_data)

        hidden_activations = np.dot(weights_input_hidden, input_data.T) + weights_bias_hidden
        hidden = sigmoid(hidden_activations)

        output_activations = np.dot(weights_hidden_output, hidden) + weights_bias_output
        output = sigmoid(output_activations)

        output_error = output_data.T - output
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
        if (epochs) % 1000 == 0:
            print(f"Epoch: {epochs}, MSE: {mse}")

        mse_history.append(mse)  # Store the current MSE in the list

    return mse_history



def mean_absolute_percentage_error(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100

def mean_absolute_error(actual, predicted):
    return np.mean(np.abs(actual - predicted))


def accuracy(actual, predicted, threshold=0.5):
    total_correct = 0
    total_samples = len(actual)

    for actual_val, predicted_val in zip(actual, predicted.T):
        predicted_binary = (predicted_val >= threshold).astype(int)
        if np.array_equal(actual_val, predicted_binary):
            total_correct += 1

    accuracy_percentage = (total_correct / total_samples) * 100
    return accuracy_percentage

def print_results(z2, zp, threshold=0.5):
    for actual_val, predicted_val in zip(z2, zp.T):
        predicted_binary = (predicted_val >= threshold).astype(int)
        is_correct = np.array_equal(actual_val, predicted_binary)
        print(f"Actual: {actual_val}, Predicted: {predicted_binary}, Correct: {is_correct}")



x , y = read_data_from_file("cross_train.txt")

init_w0(2,8,2)
mse_history = train(x, y, N = 10000 , target_mse = 0.001, lr = 0.7 , momentum_rate = 0.9 )

z1 , z2 = read_data_from_file("cross_test.txt")
zp = forward(z1)

print_results(z2, zp)
accuracy_percentage = accuracy(z2, zp)
print(f"Accuracy:", accuracy_percentage ,"%")



plt1.plot(range(1, len(mse_history) + 1), mse_history)
#plt1.xscale('log')  
plt1.xlabel('Epoch')
plt1.ylabel('Mean Squared Error (MSE)')
plt1.title('Training Progress')
plt1.grid(True)
plt1.show()
