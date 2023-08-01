import numpy as np
import matplotlib.pyplot as plt

class KohonenSOM:
    def __init__(self, input_dim, output_dim, learning_rate=0.1, num_epochs=100):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = np.random.rand(output_dim[0], output_dim[1], input_dim)

    def find_best_matching_unit(self, data_point):
        distances = np.linalg.norm(self.weights - data_point, axis=2)
        bmu_index = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_index

    def update_weights(self, data_point, bmu_index, epoch):
        radius = self.output_dim[0] / 2 * np.exp(-epoch / self.num_epochs)
        distance_matrix = self.get_distance_matrix(bmu_index)
        influence = np.exp(-distance_matrix**2 / (2 * radius**2))
        self.weights += self.learning_rate * influence[:, :, np.newaxis] * (data_point - self.weights)

    def get_distance_matrix(self, bmu_index):
        x, y = np.indices(self.output_dim)
        x_diff = x - bmu_index[0]
        y_diff = y - bmu_index[1]
        distance_matrix = np.sqrt(x_diff**2 + y_diff**2)
        return distance_matrix

    def train(self, data):
        for epoch in range(self.num_epochs):
            np.random.shuffle(data)
            for data_point in data:
                bmu_index = self.find_best_matching_unit(data_point)
                self.update_weights(data_point, bmu_index, epoch)
    
    def map_data(self, data):
        mapped_data = []
        for data_point in data:
            bmu_index = self.find_best_matching_unit(data_point)
            mapped_data.append(bmu_index)
        return mapped_data

# Example usage:
data = np.random.rand(100, 2)  # Generate some random 2D data points
som = KohonenSOM(input_dim=2, output_dim=(10, 10), learning_rate=0.1, num_epochs=100)
som.train(data)

# Map the data points to their corresponding neurons
mapped_data = som.map_data(data)

# Visualize the output
grid_size = som.output_dim[0]
neuron_positions = np.array([[x, y] for x in range(grid_size) for y in range(grid_size)])
neuron_positions_mapped = np.array([mapped_data[i] for i in range(len(mapped_data))])
neuron_positions_mapped = np.column_stack((neuron_positions_mapped, np.zeros(neuron_positions_mapped.shape[0])))

plt.figure(figsize=(6, 6))
plt.scatter(neuron_positions[:, 0], neuron_positions[:, 1], color='blue', marker='o')
plt.scatter(neuron_positions_mapped[:, 0], neuron_positions_mapped[:, 1], color='red', marker='x')
plt.title("Kohonen SOM Output")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(["Neurons", "Mapped Data Points"])
plt.grid(True)
plt.show()