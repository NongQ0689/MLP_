def decimal_to_binary(decimal):
    binary = bin(decimal)[2:]  # Convert decimal to binary string
    binary = binary.zfill(10)  # Pad the binary number with leading zeros to make it 10 bits
    return binary

def convert_numbers(input_file, output_file):
    with open(input_file, 'r') as file:
        numbers = [int(line.strip()) for line in file if line.strip().isdigit()]

    binary_numbers = [decimal_to_binary(number) for number in numbers]

    with open(output_file, 'w') as file:
        file.write('\n'.join(binary_numbers))



# Example usage
input_filename = 'Output_Only.txt'
output_filename = 'Output_Only_Bin.txt'

convert_numbers(input_filename, output_filename)
