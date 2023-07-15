

def extract_numbers(input_file):
    numbers = []
    with open(input_file, 'r') as file:
        for line in file:
            for word in line.split():
                if word.isdigit():
                    numbers.append(int(word))

    return numbers

input_file = "Data.txt"  # file name 
numbers = extract_numbers(input_file)

print(numbers, end=' ')
