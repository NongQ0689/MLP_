def extract_numbers(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()
        numbers = [line.strip() for i, line in enumerate(lines) if (i + 1) % 9 != 0 and line.strip().isdigit()]

    with open(output_file, 'w') as file:
        file.write('\n'.join(numbers))


# Example usage
input_filename = 'XData_Newline.txt'
output_filename = 'XInput_Only.txt'

extract_numbers(input_filename, output_filename)
