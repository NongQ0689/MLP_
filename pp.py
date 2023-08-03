import numpy as np

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
        if (epochs) % 1000 == 0:
            print(f"Epoch: {epochs}, MSE: {mse}")


