X = []

# Open the file in read mode
with open("Data.txt", "r") as file:
    # Read each line in the file
    for line in file:
        # Split the line by whitespace and convert each value to an integer
        values = [int(x) for x in line.split()]
        # Append the values to the list X
        X.append(values)

# Print the list X
print(X)
