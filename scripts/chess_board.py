import numpy as np

# Define the x-axis values for each column
x_values = {
    'A': 0.15,
    'B': 0.20,
    'C': 0.25
}

# Define the y-axis values
y_values = np.concatenate((np.linspace(-0.25, -0.15, 4), np.linspace(0.15, 0.25, 3)))

# Generate the chessboard positions
chessboard_positions = []
columns = 'ABC'
rows = '1234567'

for i, column in enumerate(columns):
    x_position = x_values[column]
    for j, row in enumerate(rows):
        square_name = column + row
        y_position = y_values[j]
        chessboard_positions.append((square_name, (x_position, y_position)))

# Save the positions as a list
chessboard_positions_list = [{"name": pos[0], "position": pos[1]} for pos in chessboard_positions]

# Example of how to access the list
print(chessboard_positions_list)
# target_position_name = "A1"
# target_position = get_exact_position(chessboard_positions_list, target_position_name)

# print(f"The exact position of {target_position_name} is {target_position}")

# def read_commands_from_file(file_path):
#     """
#     Reads commands from a text file and builds a task list.

#     Args:
#     file_path (str): Path to the text file containing the commands.

#     Returns:
#     list: A list of task commands.
#     """
#     task_list = []
#     with open(file_path, 'r') as file:
#         for line in file:
#             line = line.strip()
#             if line and not line.startswith('#'):
#                 command = line.split('. ', 1)[1]
#                 command_list = eval(command)
#                 task_list.append(command_list)
#     return task_list

# file_path = '/home/xz2723/niryo_project/llmAPI/task_list.txt'
# task_list = read_commands_from_file(file_path)
# print(task_list)