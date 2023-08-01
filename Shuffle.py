import random

# Function to read data from a file and return as a list of lists
def read_data_from_file(file_name):
    data = []
    with open(file_name, 'r') as file:
        for line in file:
            row = [int(num) for num in line.strip().split()]
            data.append(row)
    return data

# Function to shuffle the data
def shuffle_data(data):
    random.shuffle(data)

# Function to write data to another file
def write_data_to_file(data, file_name):
    with open(file_name, 'w') as file:
        for row in data:
            file.write(' '.join(str(num) for num in row) + '\n')

# File names
input_file_name = 'Data.txt'
output_file_name = 'shuffled_data.txt'

# Read data from the input file
data = read_data_from_file(input_file_name)

# Shuffle the data
shuffle_data(data)

# Write shuffled data to the output file
write_data_to_file(data, output_file_name)

print("Data has been read from '{}' file, shuffled, and written to '{}' file.".format(input_file_name, output_file_name))
