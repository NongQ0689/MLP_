def binary_to_array(binary):
    array = [int(bit) for bit in binary]
    return array

def extract_arrays(input_file, output_file):
    with open(input_file, 'r') as file:
        binary_numbers = [line.strip() for line in file]

    arrays = [binary_to_array(binary) for binary in binary_numbers]

    with open(output_file, 'w') as file:
        for array in arrays:
            file.write('[' + ','.join(map(str, array)) + ']\n')

# Example usage
input_filename = 'Data_Newline_Bin.txt'
output_filename = 'Data_Newline_Array.txt'

extract_arrays(input_filename, output_filename)
