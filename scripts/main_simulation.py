import math
import mujoco
import numpy as np
import time
from mujoco import viewer
import matplotlib.pyplot as plt
from pid_controller import PIDController, PIDControllerWithDerivativeFilter, AdvancedPIDController
from GradientDescentIK import GradientDescentIK
from LevenbergMarquardtIK import LevenbergMarquardtIK
from robot_controller import RobotController
import robot_controller
import simulation_steps as steps
import ball_simulation_check as check
from chess_board import chessboard_positions_list

import ast

def read_commands_from_file(file_path):
    task_list = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            # Find the position of the first '[' and the last ']' to identify the list
            start_idx = line.find('[')
            end_idx = line.rfind(']')
            if start_idx != -1 and end_idx != -1:
                list_str = line[start_idx:end_idx+1]
                try:
                    # Safely evaluate the string to a Python list
                    command_list = ast.literal_eval(list_str)
                    task_list.append(command_list)
                except (SyntaxError, ValueError) as e:
                    print(f"Skipping malformed line: {line}")
                    print(f"Error: {e}")
    return task_list

file_path = '/home/xz2723/niryo_project/llmAPI/task_list.txt'
task_list = read_commands_from_file(file_path)
print(task_list)

# get the target position using the name of the position
def get_exact_position(chessboard_positions_list, position_name):
    for position in chessboard_positions_list:
        if position["name"] == position_name:
            return position["position"][0], position["position"][1], 0.107

def get_positions_for_this_command(task_list, chessboard_positions_list):
    ball_position = None
    ball_position_name = None
    target_position = None
    target_position_name = None
    
    for task in task_list:
        if task[0] == "release":
            target_position_name = task[1]  # Corrected to task[2]
            target_position = get_exact_position(chessboard_positions_list, target_position_name)
        
        if task[0] == "grab":
            ball_position_name = task[1]  # Corrected to task[2]
            ball_position = get_exact_position(chessboard_positions_list, ball_position_name)
            
    return ball_position, ball_position_name, target_position, target_position_name

# Load the model
model_path = '/home/xz2723/niryo_project/meshes/mjmodel.xml'
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Get the positions and names of the ball and target
ball_position, ball_position_name, target_position, target_position_name = get_positions_for_this_command(task_list, chessboard_positions_list)
print("Ball position:", ball_position, "Ball position name:", ball_position_name)
print("Target position:", target_position, "Target position name:", target_position_name)

# Construct the ball body name
ball_body_name = 'ball_' + ball_position_name
ball_body_id = model.body(ball_body_name).id

print("Ball body ID:", ball_body_id, "Ball body name:", ball_body_name)

# Function to get the position of the ball
def get_ball_position(data, body_id):
    return data.body(body_id).xpos

# ball_position = get_ball_position(data, ball_body_id)
# print("Initial ball position:", ball_position)

# Joint names and initialization
joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'left_clamp_joint', 'right_clamp_joint']
joint_indices = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names}

fixed_positions = {
    'joint_1': 0.00369,
    'joint_2': -0.0135,
    'joint_3': -0.511,
    'joint_4': -1.48,
    'joint_5': -0.000,
    'joint_6': 0.901,
    'left_clamp_joint': -0.00000,
    'right_clamp_joint': 0.00036
}

initialize_angles = np.array([fixed_positions[name] for name in joint_names])

# target_position = [0.10000, 0.20000, 0.2000]

# Define the inverse kinematics class using gradient descent


# Initialize variables for inverse kinematics
body_id = model.body('g1_mainSupport_link').id
site_id = model.site('gripper_center').id
sensor_id = model.sensor('touch').id
jacp = np.zeros((3, model.nv))
jacr = np.zeros((3, model.nv))
step_size = 0.01
tol = 0.001
alpha = 0.5
init_q = initialize_angles

pid_ball = PIDController(100, 0.5, 100)

movable_joints_indices = [joint_indices[joint_name] for joint_name in joint_names[:]]  # Exclude clamp joints

# Initialize PID controllers
pids = {
    'joint_1': PIDController(100, 0, 100),
    'joint_2': PIDController(100, 0, 100),
    'joint_3': PIDController(100, 0, 100),
    'joint_4': PIDController(180, 0, 100),
    'joint_5': PIDController(100, 0, 100),
    'joint_6': PIDController(162.5, 0, 100),
    'left_clamp_joint': PIDControllerWithDerivativeFilter(20, 0, 100),
    'right_clamp_joint': PIDControllerWithDerivativeFilter(20, 0, 100)
}

joint_angle_history = {name: [] for name in joint_names}
position_updates = []

FLAG = 0

def get_target_angles(model, data, target_position, initialize_angles, body_id, jacp, jacr, movable_joints_indices):
    ik = GradientDescentIK(model, data, step_size, tol, alpha, jacp, jacr, movable_joints_indices)
    ik_lm = LevenbergMarquardtIK(model, data, step_size, tol, alpha, 0.1, movable_joints_indices)    
    target_angles = ik_lm.calculate(target_position, initialize_angles[:8], body_id)
    print("Target angles:", target_angles)
    return target_angles

def pre_calc(model, data, target_position, initialize_angles, body_id, jacp, jacr, movable_joints_indices):
    print("Calculating target angles...")
    target_angles = get_target_angles(model, data, target_position, initialize_angles, body_id, jacp, jacr, movable_joints_indices)
    print("Target angles:", target_angles)
    return target_angles

########### define different tasks ###########
niryo = RobotController(model, data, joint_names, pids, joint_indices=joint_indices)

def move(steps, niryo, target_angles, FLAG, joint_indices, data, site_id, body_id, ball_body_name):
    print("ball_name:", ball_body_name)
    for step in range(steps):
        target_angles = check.updates_gripper_angles(niryo, target_angles, FLAG, joint_indices)
        RobotController.control_arm_to_position(niryo, target_angles)
        
        if FLAG == 1:
            ball_pos = check.ball_pos_update(data, site_id)
            RobotController.set_mocap_position(niryo, ball_body_name, ball_pos)
            Viewer.sync()
        
        RobotController.sync_viewer(niryo, Viewer)
        time.sleep(0.02)
    
    return FLAG

def close_the_gripper(steps, niryo, target_angles, FLAG, joint_indices, data, site_id, body_id, ball_body_name):
    print("task: close the gripper")
    FLAG = 1
    print("ball_name:", ball_body_name)
    for step in range(steps):
        target_angles = check.updates_gripper_angles(niryo, target_angles, FLAG, joint_indices)
        RobotController.control_arm_to_position(niryo, target_angles)
        if check.check_before_close_gripper(data, body_id, target_position, 0.01):
            FLAG = 1
            # print("grab the ball")
        FLAG = 1
        if FLAG == 1:
            ball_pos = check.ball_pos_update(data, site_id)
            RobotController.set_mocap_position(niryo, ball_body_name, ball_pos)
            Viewer.sync()
        
        RobotController.sync_viewer(niryo, Viewer)
        time.sleep(0.02)
    
    return FLAG

def open_the_gripper(steps, niryo, target_angles, FLAG, joint_indices, data, site_id, body_id, ball_body_name):
    print("task: open the gripper")
    FLAG = 0
    for step in range(steps):
        target_angles = check.updates_gripper_angles(niryo, target_angles, FLAG, joint_indices)
        RobotController.control_arm_to_position(niryo, target_angles)
        if FLAG == 1:
            ball_pos = check.ball_pos_update(data, site_id)
            RobotController.set_mocap_position(niryo, ball_body_name, ball_pos)
            Viewer.sync()
        else:
            ball_pos = target_position
            RobotController.set_mocap_position(niryo, ball_body_name, ball_pos)
            Viewer.sync()
        
        RobotController.sync_viewer(niryo, Viewer)
        time.sleep(0.02)
    
    return FLAG

#################################################

################# get target angles in different stages #################
def get_task_name_and_target_angles(model, data, ball_position, target_position, initialize_angles, body_id, jacp, jacr, movable_joints_indices, task_name, ball_position_name, target_position_name):
    # if the task name is a array like [move, lower above, ball position, none], [grab, ball, none, none], [release, ball, none, none], get the main token for further calculation
    if task_name[0] == "move" and task_name[1] == "lower above" and task_name[2] == ball_position_name:
        if ball_position[0] >= 0:
            target_position_for_move = ball_position + np.array([-0.008, -0.001, 0])
        target_position_for_move = ball_position + np.array([0, 0, 0.12])
    elif task_name[0] == "move" and task_name[1] == "higher above" and task_name[2] == ball_position_name:
        target_position_for_move = ball_position + np.array([0, 0, 0.18])
    elif task_name[0] == "move" and task_name[1] == "exact" and task_name[2] == ball_position_name:
        if ball_position[0] >= 0:
            target_position_for_move = ball_position + np.array([-0.008, -0.001, 0])
        target_position_for_move = ball_position + np.array([0, 0, 0.04])

    elif task_name[0] == "move" and task_name[1] == "lower above" and task_name[2] == target_position_name:
        if target_position[0] >= 0:
            target_position_for_move = target_position + np.array([-0.008, -0.001, 0])
        target_position_for_move = target_position + np.array([0, 0, 0.12])
    elif task_name[0] == "move" and task_name[1] == "higher above" and task_name[2] == target_position_name:
        target_position_for_move = target_position + np.array([0, 0, 0.18])
    elif task_name[0] == "move" and task_name[1] == "exact" and task_name[2] == target_position_name:
        if target_position[0] >= 0:
            target_position_for_move = target_position + np.array([-0.008, -0.001, 0])
        target_position_for_move = target_position + np.array([0, 0, 0.04])
    
    elif task_name[0] == "grab":
        if ball_position[0] >= 0:
            target_position_for_move = ball_position + np.array([-0.005, -0.001, 0])
        target_position_for_move = ball_position + np.array([0, 0, 0.07])
    elif task_name[0] == "release":
        if target_position[0] >= 0:
            target_position_for_move = target_position + np.array([-0.005, -0.001, 0])
        target_position_for_move = target_position + np.array([0, 0, 0.07])
    
    elif task_name[2] == "initial position":
        target_position_for_move = np.array([0.23, 0.00, 0.20])
    
    target_angles = pre_calc(model, data, target_position_for_move, initialize_angles, body_id, jacp, jacr, movable_joints_indices)
    print(task_name, "target angles:", target_angles)
    return target_angles


def control_command(steps, niryo, target_angles, FLAG, joint_indices, data, site_id, body_id, task_name, ball_position_name, target_position_name, ball_body_name):
    # target_angles = get_task_name_and_target_angles(model, data, ball_position, target_position, initialize_angles, body_id, jacp, jacr, movable_joints_indices, task_name)
    
    if task_name[0] == "move" and task_name[1] == "lower above" and task_name[2] == ball_position_name:
        print("moving to the lower above of ball position")
        FLAG = move(steps, niryo, target_angles, FLAG, joint_indices, data, site_id, body_id, ball_body_name)
    elif task_name[0] == "move" and task_name[1] == "higher above" and task_name[2] == ball_position_name:
        print("lift the ball to higher above position")
        FLAG = move(steps, niryo, target_angles, FLAG, joint_indices, data, site_id, body_id, ball_body_name)
    elif task_name[0] == "move" and task_name[1] == "exact" and task_name[2] == ball_position_name:
        print("moving down to the ball position")
        FLAG = move(steps, niryo, target_angles, FLAG, joint_indices, data, site_id, body_id, ball_body_name)

    elif task_name[0] == "grab":
        print("close the gripper\n")
        FLAG = close_the_gripper(steps, niryo, target_angles, FLAG, joint_indices, data, site_id, body_id, ball_body_name)
    elif task_name[0] == "release":
        print("release the gripper")
        FLAG = open_the_gripper(steps, niryo, target_angles, FLAG, joint_indices, data, site_id, body_id, ball_body_name)

    elif task_name[0] == "move" and task_name[1] == "lower above" and task_name[2] == target_position_name:
        print("move down to the lower above of target position")
        FLAG = move(steps, niryo, target_angles, FLAG, joint_indices, data, site_id, body_id, ball_body_name)
    elif task_name[0] == "move" and task_name[1] == "exact" and task_name[2] == target_position_name:
        print("move to the target position")
        FLAG = move(steps, niryo, target_angles, FLAG, joint_indices, data, site_id, body_id, ball_body_name)
    elif task_name[0] == "move" and task_name[1] == "higher above" and task_name[2] == target_position_name:
        print("move to the higher above of target position")
        FLAG = move(steps, niryo, target_angles, FLAG, joint_indices, data, site_id, body_id, ball_body_name)

    elif task_name[0] == "move" and task_name[1] == "initial position":
        print("move to the initial position")
        FLAG = move(steps, niryo, target_angles, FLAG, joint_indices, data, site_id, body_id, ball_body_name)

    return FLAG

# Simulation loop for synchronized movement
with viewer.launch_passive(model, data) as Viewer:
    
    Viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 0
    Viewer.sync()
    # ball_position = get_ball_position(data, ball_body_id)
    # print("Initial ball position:", ball_position)
    # 0.23, -0.16, 0.107
    
    ball_position, ball_position_name, target_position, target_position_name = get_positions_for_this_command(task_list, chessboard_positions_list)
    print("Ball position:", ball_position, "Ball position name:", ball_position_name)
    print("Target position:", target_position, "Target position name:", target_position_name)
    # print("Target position:", target_position)
    
    # ball body name
    ball_body_name = 'ball_' + ball_position_name
    print("Ball body name:", ball_body_name)
    
    '''
    The steps are:
    1 moving to the position that slightly above the target object, the distance above teh object is fixed
    2 move downward to the object, so it stays at the position of the end effector
    3 close the gripper
    4 move upward a little to avoid collision
    5 move to the above position of target position, only change the x and y axis position, keep the height the same as the previous step
    6 move to the slightly above target position point, use the same height as step1 fixed height
    7 move downward so the end effector and object at the same position
    8 release the gripper
    9 move up the robot arm to release the ball
    
    '''

    # Simulation parameters
    duration = 5  # seconds
    steps_1 = int(duration * 50)
    steps_2 = int(duration * 50)
    steps_3 = int(duration * 50)
    steps_4 = int(duration * 50)
    steps_5 = int(duration * 50)
    steps_6 = int(duration * 50)
    Steps = int(duration * 50)

    # Set initial joint positions
    for name, angle in fixed_positions.items():
        joint_index = joint_indices[name]
        data.qpos[joint_index] = angle

    Viewer.sync()
    time.sleep(0.02)
    
    angle_list = []
    
    print("calculating angle values...")
    print("Task list:", task_list)
    for task_name in task_list:
        target_angles = get_task_name_and_target_angles(model, data, ball_position, target_position, initialize_angles, body_id, jacp, jacr, movable_joints_indices, task_name, ball_position_name, target_position_name)
        print("Target angles:", target_angles)
        angle_list.append(target_angles)

    print("Starting simulation...")

    for task_name in task_list:
        print("Task name:", task_name)
        # breakpoint()
        target_angles = angle_list.pop(0)
        print("Target angles", target_angles)
        FLAG = control_command(Steps, niryo, target_angles, FLAG, joint_indices, data, site_id, body_id, task_name, ball_position_name, target_position_name, ball_body_name)
        Viewer.sync()
        time.sleep(1)
    
    print("Simulation finished.")
