import re

def extract_numbers(input_file, output_file):
    with open(input_file, 'r') as file:
        data = file.read()
        numbers = re.findall(r'\d+', data)  # Extract all numbers using regular expression
    
    with open(output_file, 'w') as file:
        for number in numbers:
            file.write(number + '\n')  # Write each number to a new line in the output file

# Example usage
input_file_path = 'Xinput.txt'
output_file_path = 'Xoutput.txt'
extract_numbers(input_file_path, output_file_path)
