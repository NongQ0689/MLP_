#include <iostream>
#include <cmath>
#include <vector>

// Activation function - Sigmoid
double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

// Derivative of the activation function
double sigmoid_derivative(double x) {
    return x * (1 - x);
}

// MLP class
class MLP {
private:
    int input_size;
    int hidden_size;
    int output_size;
    std::vector<std::vector<double>> weights1;
    std::vector<std::vector<double>> weights2;

public:
    MLP(int input, int hidden, int output) {
        input_size = input;
        hidden_size = hidden;
        output_size = output;

        // Initialize weights with random values
        weights1.resize(input_size, std::vector<double>(hidden_size));
        weights2.resize(hidden_size, std::vector<double>(output_size));

        // Initialize weights randomly
        for (int i = 0; i < input_size; i++) {
            for (int j = 0; j < hidden_size; j++) {
                weights1[i][j] = (double)rand() / RAND_MAX;
            }
        }

        for (int i = 0; i < hidden_size; i++) {
            for (int j = 0; j < output_size; j++) {
                weights2[i][j] = (double)rand() / RAND_MAX;
            }
        }
    }

    std::vector<double> forward(const std::vector<double>& X) {
        std::vector<double> hidden_layer(hidden_size);
        std::vector<double> output_layer(output_size);

        // Forward propagation through the network
        for (int i = 0; i < hidden_size; i++) {
            double sum = 0;
            for (int j = 0; j < input_size; j++) {
                sum += X[j] * weights1[j][i];
            }
            hidden_layer[i] = sigmoid(sum);
        }

        for (int i = 0; i < output_size; i++) {
            double sum = 0;
            for (int j = 0; j < hidden_size; j++) {
                sum += hidden_layer[j] * weights2[j][i];
            }
            output_layer[i] = sigmoid(sum);
        }

        return output_layer;
    }

    void backward(const std::vector<double>& X, const std::vector<double>& y, double learning_rate) {
        std::vector<double> hidden_layer(hidden_size);
        std::vector<double> output_layer(output_size);

        // Forward propagation
        hidden_layer = forward(X);

        // Backward propagation and weight update
        std::vector<double> delta_output(output_size);
        std::vector<double> delta_hidden(hidden_size);

        for (int i = 0; i < output_size; i++) {
            double error = y[i] - hidden_layer[i];
            delta_output[i] = error * sigmoid_derivative(hidden_layer[i]);
        }

        for (int i = 0; i < hidden_size; i++) {
            double error = 0;
            for (int j = 0; j < output_size; j++) {
                error += delta_output[j] * weights2[i][j];
            }
            delta_hidden[i] = error * sigmoid_derivative(hidden_layer[i]);
        }

        for (int i = 0; i < input_size; i++) {
            for (int j = 0; j < hidden_size; j++) {
                weights1[i][j] += X[i] * delta_hidden[j] * learning_rate;
            }
        }

        for (int i = 0; i < hidden_size; i++) {
            for (int j = 0; j < output_size; j++) {
                weights2[i][j] += hidden_layer[i] * delta_output[j] * learning_rate;
            }
        }
    }

    void train(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& y, int epochs, double learning_rate) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < X.size(); i++) {
                forward(X[i]);
                backward(X[i], y[i], learning_rate);
            }
        }
    }

    std::vector<double> predict(const std::vector<double>& X) {
        return forward(X);
    }
};

int main() {
    std::vector<std::vector<double>> X = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
    std::vector<std::vector<double>> y = { {0}, {1}, {1}, {0} };

    MLP mlp(2, 4, 1);  // Create an MLP with 2 input nodes, 4 hidden nodes, and 1 output node
    mlp.train(X, y, 10000, 0.1);  // Train the MLP

    // Make predictions
    for (const auto& input : X) {
        std::vector<double> predictions = mlp.predict(input);
        for (const auto& pred : predictions) {
            std::cout << pred << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
