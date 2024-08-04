# Define the bounding box of the chessboard
top_left = (0.20, 0.15)
top_right = (0.30, 0.15)
bottom_left = (0.20, -0.15)
bottom_right = (0.30, -0.15)

# Calculate the width and height of each square
board_width = top_right[0] - top_left[0]
board_height = top_left[1] - bottom_left[1]
square_size_x = board_width / 8
square_size_y = board_height / 8

# Generate the chessboard positions
chessboard_positions = []
columns = 'ABCDEFGH'
rows = '12345678'

for i in range(8):
    for j in range(8):
        square_name = columns[i] + rows[j]
        x_position = top_left[0] + i * square_size_x + square_size_x / 2
        y_position = top_left[1] - j * square_size_y - square_size_y / 2
        chessboard_positions.append((square_name, (x_position, y_position)))

# Print the chessboard positions
# for position in chessboard_positions:
    # print(f"{position[0]} is at position {position[1]}")

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