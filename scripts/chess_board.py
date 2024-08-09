import numpy as np

# Define the x-axis values for each column
x_values = {
    'A': 0.175,
    'B': 0.225,
    'C': 0.275
}

# Define the y-axis values for each row
y_values = {
    '1': -0.195,
    '2': -0.145,
    '3': -0.095,
    '4': -0.045,
    '5': 0.005,
    '6': 0.055,
    '7': 0.105
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
