def extract_numbers(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    numbers = []
    for i in range(8, len(lines), 9):
        line = lines[i].strip()
        if line.isdigit():
            numbers.append(line)

    with open(output_file, 'w') as file:
        file.write('\n'.join(numbers))


# Example usage
input_filename = 'Xoutput.txt'
output_filename = 'Xoutput3.txt'

extract_numbers(input_filename, output_filename)
