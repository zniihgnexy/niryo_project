import numpy as np

# Define the x-axis values for each column
x_values = {
    'A': 0.180,
    'B': 0.235,
    'C': 0.28
}

# Define the y-axis values for each row
y_values = {
    "1": -0.145,
    "2": -0.105,
    "3": -0.065,
    "4": -0.025,
    "5": 0.02,
    "6": 0.06,
    "7": 0.10,
    "8": 0.145
}

# Generate the chessboard positions
chessboard_positions_list = []

# Iterate over each column and row combination
for column in x_values.keys():
    x_position = x_values[column]
    for row in y_values.keys():
        square_name = column + row
        y_position = y_values[row]
        chessboard_positions_list.append({"name": square_name, "position": (x_position, y_position)})

# Example of how to print the generated list
print(chessboard_positions_list)
