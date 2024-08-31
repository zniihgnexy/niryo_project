import numpy as np
"""
Generate chessboard positions based on x and y values.

Parameters:
    None

Returns:
    chessboard_positions_list (list): A list of dictionaries containing the name and position of each square on the chessboard.

Example:
    chessboard_positions_list = generate_chessboard_positions()
    Output:
    [{'name': 'A1', 'position': (0.18, -0.145)}, {'name': 'A2', 'position': (0.18, -0.105)}, {'name': 'A3', 'position': (0.18, -0.065)}, {'name': 'A4', 'position': (0.18, -0.025)}, {'name': 'A5', 'position': (0.18, 0.02)}, {'name': 'A6', 'position': (0.18, 0.06)}, {'name': 'A7', 'position': (0.18, 0.1)}, {'name': 'A8', 'position': (0.18, 0.145)}, {'name': 'B1', 'position': (0.235, -0.145)}, {'name': 'B2', 'position': (0.235, -0.105)}, {'name': 'B3', 'position': (0.235, -0.065)}, {'name': 'B4', 'position': (0.235, -0.025)}, {'name': 'B5', 'position': (0.235, 0.02)}, {'name': 'B6', 'position': (0.235, 0.06)}, {'name': 'B7', 'position': (0.235, 0.1)}, {'name': 'B8', 'position': (0.235, 0.145)}, {'name': 'C1', 'position': (0.28, -0.145)}, {'name': 'C2', 'position': (0.28, -0.105)}, {'name': 'C3', 'position': (0.28, -0.065)}, {'name': 'C4', 'position': (0.28, -0.025)}, {'name': 'C5', 'position': (0.28, 0.02)}, {'name': 'C6', 'position': (0.28, 0.06)}, {'name': 'C7', 'position': (0.28, 0.1)}, {'name': 'C8', 'position': (0.28, 0.145)}]
"""

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
