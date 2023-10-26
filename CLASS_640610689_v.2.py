import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

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


def shuffle_data(input, output):
    # Combine 'input' and 'output' into a list of tuples
    data = list(zip(input, output))

    # Shuffle the data using numpy's random.shuffle
    np.random.shuffle(data)

    # Split the shuffled data back into 'input' and 'output' arrays
    input, output = zip(*data)
    input = list(input)
    output = list(output)

    return input, output

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

def forward(input_data ):
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
        #if (epochs) % 10000 == 0:
        #    print(f"Epoch: {epochs}, MSE: {mse:.4f}")

        mse_history.append(mse)  # Store the current MSE in the list
    last_mse = mse
    return mse_history , last_mse


##################################################  MAPE  MAE:

def mean_absolute_percentage_error(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100

def mean_absolute_error(actual, predicted):
    return np.mean(np.abs(actual - predicted)) 

##################################################  accuracy  print_results confusion_matrix:

def accuracy(actual, predicted, threshold=0.5):
    total_correct = 0
    total_samples = len(actual)

    for actual_val, predicted_val in zip(actual, predicted.T):
        predicted_binary = (predicted_val >= threshold).astype(int)
        if np.array_equal(actual_val, predicted_binary):
            total_correct += 1

    accuracy_percentage = (total_correct / total_samples) * 100
    return accuracy_percentage

def print_results(actual, predicted, threshold=0.5):
    correct_count = 0
    total_count = len(actual)

    TP, TN, FP, FN = 0, 0, 0, 0
    correctness = []

    for actual_val, predicted_val in zip(actual, predicted.T):
        predicted_binary = (predicted_val >= threshold).astype(int)
        is_correct = np.array_equal(actual_val, predicted_binary)
        correctness.append(is_correct)

        if is_correct:
            correct_count += 1

        # Update TP, TN, FP, FN based on the actual and predicted values
        if actual_val[0] == 1 and predicted_binary[0] == 1:
            TP += 1
        elif actual_val[0] == 0 and predicted_binary[0] == 0:
            TN += 1
        elif actual_val[0] == 0 and predicted_binary[0] == 1:
            FP += 1
        elif actual_val[0] == 1 and predicted_binary[0] == 0:
            FN += 1

        print(f"Actual: [{int(actual_val[0])} {int(actual_val[1])}] , Predicted: {predicted_binary}, Correct: {is_correct}")

    print("...")
    accuracy_percentage = (correct_count / total_count) * 100
    print(f"Accuracy: {accuracy_percentage:.2f}% ({correct_count}/{total_count} correct predictions)")

    return TP, TN, FP, FN

def confusion_matrix(TP, TN, FP, FN):
    confusion_matrix = [[TN, FP], [FN, TP]]

    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = [0, 1]
    plt.xticks(tick_marks, ["Predicted [0 1]", "Predicted [1 0]"])
    plt.yticks(tick_marks, ["Actual [0 1]", "Actual [1 0]"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(confusion_matrix[i][j]), horizontalalignment="center", color="white" if confusion_matrix[i][j] > (TP + TN + FP + FN) / 2 else "black")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

##################################################  cross_validation

def k_fold_data(k, data,type):
    folds = []
    fold_size = len(data)//k

    for i in range(fold_size):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size
        if type == "test":
            t_fold = data[start_idx:end_idx]
        if type == "train":
            t_fold = data[:start_idx] + data[end_idx:]

        folds.append(t_fold)

    return folds 

def k_fold_cross_validation(k, input_size, hidden_size, output_size, N, target_mse, lr, momentum_rate):
   
    input_train = k_fold_data(k, input,"train")
    output_train = k_fold_data(k, output,"train")
    input_test = k_fold_data(k, input,"test")
    output_test = k_fold_data(k, output,"test")

    best_accuracy = 0
    best_accuracy_i = 0
    best_mse = 1

    best_weight = []

    for k_fold in range(k) :
        train_X, train_Y = input_train[k_fold] , output_train[k_fold]
        test_X, test_Y = input_test[k_fold] , output_test[k_fold]

        init_w0(input_size, hidden_size, output_size)
        mse_history , last_mse = train(train_X, train_Y, N, target_mse, lr, momentum_rate)
        prediction = forward(test_X)
        #denormalize_predictions = denormalize_data(prediction, minimum, maximum)
        #mape = mean_absolute_percentage_error(test_Y, prediction)
        #mae = mean_absolute_error(test_Y, prediction)
        Accuracy = accuracy(test_Y, prediction, threshold=0.5)
        

        if Accuracy > best_accuracy:
            actual_outputs = test_Y
            predicted_outputs = prediction
            best_accuracy_i = k_fold
            best_weight = [weights_input_hidden, weights_hidden_output, weights_bias_hidden, weights_bias_output]

            best_accuracy = Accuracy #update best_accuracy
        
        if last_mse < best_mse:
            best_mse_history = mse_history
            best_mse_i = k_fold
            best_mse = last_mse

        print(f"{k_fold}: Test_Accuracy: {Accuracy:.2f}% , MSE: {last_mse:.5f}")
        print("...")

    print(f"Best_accuracy : {best_accuracy_i} : {best_accuracy:.2f}%")
    print(f"Best_mse : {best_mse_i} : {best_mse:.5f}%")
    print("...")

    return best_mse_history , actual_outputs , predicted_outputs , best_weight

    

##################################################

k=10
input_size = 2
hidden_size = 4
output_size = 2
N = 1000
target_mse = 0.0001
lr = 0.1
momentum_rate = 0.1

##################################################

initw0 = False
input , output = read_data_from_file("cross.pat")
input, output = shuffle_data(input, output) # Shuffle the data using the function

best_mse_history , actual_outputs , predicted_outputs ,best_weight= k_fold_cross_validation(k, input_size, hidden_size, output_size, N, target_mse, lr, momentum_rate)

predicted = forward_out(input,best_weight)
actual = output
TP, TN, FP, FN = print_results(actual, predicted)


plt.figure(figsize=(10, 6))
plt.plot(best_mse_history, label='MSE')
plt.title('Mean Squared Error (MSE) Over Epochs')
#plt.xscale('log')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.show()

confusion_matrix(TP, TN, FP, FN)
