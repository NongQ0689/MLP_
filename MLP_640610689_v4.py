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

def denormalize_single_value(value, min_value, max_value):
    denormalized_value = (value * (max_value - min_value)) + min_value
    return denormalized_value

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
    global weights_input_hidden, weights_hidden_output, weights_bias_hidden, weights_bias_output,initw0
    global weights_input_hidden_0, weights_hidden_output_0, weights_bias_hidden_0, weights_bias_output_0
    
    if initw0 == False:
        print("start")
        weights_input_hidden_0 = np.random.randn(hidden_size, input_size)
        weights_hidden_output_0 = np.random.randn(output_size, hidden_size)
        weights_bias_hidden_0 = np.random.randn(hidden_size, 1)
        weights_bias_output_0 = np.random.randn(output_size, 1)
        initw0 = True
    
    weights_input_hidden = weights_input_hidden_0
    weights_hidden_output = weights_hidden_output_0
    weights_bias_hidden = weights_bias_hidden_0
    weights_bias_output = weights_bias_output_0

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

def forward_out(input_data , best_weight):
    weights_input_hidden, weights_hidden_output, weights_bias_hidden, weights_bias_output = best_weight
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
        #if (epochs) % 100000 == 0:
        #    print(f"Epoch: {epochs}, MSE: {mse}")

        mse_history.append(mse)  # Store the current MSE in the list
    last_mse = mse

    return mse_history , last_mse

##################################################  MAPE  MAE:

def mean_absolute_percentage_error(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100

def mean_absolute_error(actual, predicted):
    return np.mean(np.abs(actual - predicted)) 

def print_results(actual, predicted):
    mape = mean_absolute_percentage_error(actual, predicted)
    accuracy = 100 - mape
    
    N_input = len(actual)

    print(f"Prediction_data: {N_input} , Prediction_accuracy: {accuracy:.4f}% ")
    print("...")


##################################################  cross_validation


def k_fold_data(k, data):
    folds = []
    fold_size = len(data) // k

    for i in range(k):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size
        test_fold = data[start_idx:end_idx]
        train_fold = data[:start_idx] + data[end_idx:]
        folds.append((train_fold, test_fold))
    
    return folds

def k_fold_cross_validation(k, input_size, hidden_size, output_size, N, target_mse, lr, momentum_rate):

    data = read_data_from_file("Data.txt") ############################################################# file
    np.random.shuffle(data)
    minimum, maximum = min_max(data)
    normalized_data = normalize_data(data, minimum, maximum)


    actual_outputs = []  # To store actual outputs
    predicted_outputs = []  # To store predicted outputs

    best_mse_history = []
    best_accuracy = 0
    best_accuracy_i = 0
    best_weight = []
    best_mse = 1
    
    i = k
    for train_data, test_data in k_fold_data(k, normalized_data):
        train_X, train_Y = extract_features_labels(train_data)
        test_X, test_Y = extract_features_labels(test_data)

        init_w0(input_size, hidden_size, output_size)
        mse_history , last_mse = train(train_X, train_Y, N, target_mse, lr, momentum_rate)

        prediction = forward(test_X)
        #denormalize_predictions = denormalize_data(prediction, minimum, maximum)
        mape = mean_absolute_percentage_error(test_Y, prediction)
        mae = mean_absolute_error(test_Y, prediction)
        accuracy = 100 - mape
        i-=1

        if accuracy > best_accuracy:
            actual_outputs = test_Y
            predicted_outputs = prediction
            best_accuracy = accuracy
            best_accuracy_i = 10-i
            best_weight = [weights_input_hidden, weights_hidden_output, weights_bias_hidden, weights_bias_output]

        if last_mse < best_mse:
            best_mse_history = mse_history
            best_mse_i = 10-i
            best_mse = last_mse

        print(f"{10-i}: Test_Accuracy: {accuracy:.3f}%")
        print(f"{10-i}: Mean Square Error (MSE): {last_mse:.5f}%")
        print(f"{10-i}: Mean Absolute Percentage Error (MAPE): {mape:.3f}%")
        print(f"{10-i}: Mean Absolute Error (MAE): {mae:.3f}")
        print("...")

    print(f"Best_accuracy : {best_accuracy_i} : {best_accuracy:.3f}%")
    print(f"Best_MSE : {best_mse_i} : {best_mse:.5f}%")
    print("...")

    return best_mse_history , actual_outputs , predicted_outputs , minimum , maximum , best_weight
    
##################################################  train_MLP

k = 10
input_size = 8
hidden_size = 16
output_size = 1
N = 10000
target_mse = 0.0001
lr = 0.1
momentum_rate = 0.1

initw0 = False
best_mse_history , actual_outputs , predicted_outputs, minimum , maximum , best_weight = k_fold_cross_validation(k, input_size, hidden_size, output_size, N, target_mse, lr, momentum_rate)

##################################################  output use the best weight


prediction_data = read_data_from_file("Data.txt") # input output
minimum, maximum = min_max(prediction_data)
normalized_data = normalize_data(prediction_data, minimum, maximum) # normalize_data input output
prediction_data_input, prediction_data_output = extract_features_labels(normalized_data) # extract_data

predicted_outputs = forward_out(prediction_data_input,best_weight) # predicted_outputs
actual_outputs = prediction_data_output # actual_outputs

print_results(actual_outputs, predicted_outputs)

################################################## denormalized_actual_output & predicted_output

denormalized_actual_output = [denormalize_single_value(value, minimum, maximum) for value in actual_outputs]
denormalized_predicted_output = [denormalize_single_value(value, minimum, maximum)for value in predicted_outputs]
denormalized_predicted_output = denormalized_predicted_output[0].tolist()

################################################## graph MSE Over Epochs

plt.figure(figsize=(10, 6))
plt.plot(best_mse_history, label='MSE')
plt.title('Mean Squared Error (MSE) Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.show()

################################################## bar graph Actual vs. Predicted Values

categories = np.arange(len(denormalized_actual_output))
bar_width = 0.4

# Create the actual and predicted bars side by side
plt.bar(categories - bar_width/2, denormalized_actual_output, bar_width, label='Desire Output', color='b')
plt.bar(categories + bar_width/2, denormalized_predicted_output, bar_width, label='Predicted Output', color='r')

# Set x-axis labels and title
plt.xlabel('Data_set')
plt.ylabel('Water_Level')
plt.title('Desire Output vs. Predicted Output')

plt.legend()
plt.show()
