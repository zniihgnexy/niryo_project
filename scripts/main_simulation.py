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

# Load the model
model_path = '/home/xz2723/niryo_project/meshes/mjmodel.xml'
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Get the ID of the ball body
ball_body_id = model.body('box').id

# Function to get the position of the ball
def get_ball_position(data, body_id):
    return data.body(body_id).xpos

ball_position = get_ball_position(data, ball_body_id)
print("Initial box position:", ball_position)

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

def move(steps, niryo, target_angles, FLAG, joint_indices, data, site_id, body_id):
    for step in range(steps):
        target_angles = check.updates_gripper_angles(niryo, target_angles, FLAG, joint_indices)
        RobotController.control_arm_to_position(niryo, target_angles)
        
        if FLAG == 1:
            ball_pos = check.ball_pos_update(data, site_id)
            RobotController.set_mocap_position(niryo, "box", ball_pos)
            Viewer.sync()
        
        RobotController.sync_viewer(niryo, Viewer)
        time.sleep(0.02)
    
    return FLAG

def close_the_gripper(steps, niryo, target_angles, FLAG, joint_indices, data, site_id, body_id):
    print("task: close the gripper")
    FLAG = 1
    for step in range(steps):
        target_angles = check.updates_gripper_angles(niryo, target_angles, FLAG, joint_indices)
        RobotController.control_arm_to_position(niryo, target_angles)
        if check.check_before_close_gripper(data, body_id, target_position, 0.01):
            FLAG = 1
            # print("grab the ball")
        if FLAG == 1:
            ball_pos = check.ball_pos_update(data, site_id)
            RobotController.set_mocap_position(niryo, "box", ball_pos)
            Viewer.sync()
        
        RobotController.sync_viewer(niryo, Viewer)
        time.sleep(0.02)
    
    return FLAG

def open_the_gripper(steps, niryo, target_angles, FLAG, joint_indices, data, site_id, body_id):
    print("task: open the gripper")
    FLAG = 0
    for step in range(steps):
        target_angles = check.updates_gripper_angles(niryo, target_angles, FLAG, joint_indices)
        RobotController.control_arm_to_position(niryo, target_angles)
        if FLAG == 1:
            ball_pos = check.ball_pos_update(data, site_id)
            RobotController.set_mocap_position(niryo, "box", ball_pos)
            Viewer.sync()
        else:
            ball_pos = target_position
            RobotController.set_mocap_position(niryo, "box", ball_pos)
            Viewer.sync()
        
        RobotController.sync_viewer(niryo, Viewer)
        time.sleep(0.02)
    
    return FLAG

#################################################

################# get target angles in different stages #################
def get_task_name_and_target_angles(model, data, ball_position, target_position, initialize_angles, body_id, jacp, jacr, movable_joints_indices, task_name):
    # if the task name is a array like [move, lower above, ball position, none], [grab, ball, none, none], [release, ball, none, none], get the main token for further calculation
    if task_name[0] == "move" and task_name[1] == "lower above" and task_name[2] == "ball position":
        if ball_position[0] >= 0:
            target_position_for_move = ball_position + np.array([-0.006, 0, 0])
        target_position_for_move = ball_position + np.array([0, 0, 0.12])
    elif task_name[0] == "move" and task_name[1] == "higher above" and task_name[2] == "ball position":
        target_position_for_move = ball_position + np.array([0, 0, 0.18])
    elif task_name[0] == "move" and task_name[1] == "exact" and task_name[2] == "ball position":
        if ball_position[0] >= 0:
            target_position_for_move = ball_position + np.array([-0.006, 0, 0])
        target_position_for_move = ball_position + np.array([0, 0, 0.06])

    elif task_name[0] == "move" and task_name[1] == "lower above" and task_name[2] == "target position":
        if target_position[0] >= 0:
            target_position_for_move = target_position + np.array([-0.006, 0, 0])
        target_position_for_move = target_position + np.array([0, 0, 0.12])
    elif task_name[0] == "move" and task_name[1] == "higher above" and task_name[2] == "target position":
        target_position_for_move = target_position + np.array([0, 0, 0.18])
    elif task_name[0] == "move" and task_name[1] == "exact" and task_name[2] == "target position":
        if target_position[0] >= 0:
            target_position_for_move = target_position + np.array([-0.006, 0, 0])
        target_position_for_move = target_position + np.array([0, 0, 0.06])
    
    elif task_name[0] == "grab" and task_name[1] == "ball":
        if ball_position[0] >= 0:
            target_position_for_move = ball_position + np.array([-0.005, 0, 0])
        target_position_for_move = ball_position + np.array([0, 0, 0.07])
    elif task_name[0] == "release" and task_name[1] == "ball":
        if target_position[0] >= 0:
            target_position_for_move = target_position + np.array([-0.005, 0, 0])
        target_position_for_move = target_position + np.array([0, 0, 0.07])
    
    elif task_name[0] == "return" and task_name[1] == "initial position":
        target_position_for_move = np.array([0.23, 0.00, 0.20])
    
    target_angles = pre_calc(model, data, target_position_for_move, initialize_angles, body_id, jacp, jacr, movable_joints_indices)
    print(task_name, "target angles:", target_angles)
    return target_angles
    

task_list = [
    ["move", "lower above", "ball position", "none"],
    ["move", "exact", "ball position", "none"],
    ["grab", "ball", "none", "none"],
    ["move", "higher above", "ball position", "none"],
    ["move", "higher above", "target position", "none"],
    ["move", "lower above", "target position", "none"],
    ["move", "exact", "target position", "none"],
    ["release", "ball", "none", "none"],
    ["move", "higher above", "target position", "none"],
    ["return", "initial position", "none", "none"]
]

def control_command(steps, niryo, target_angles, FLAG, joint_indices, data, site_id, body_id, task_name):
    # target_angles = get_task_name_and_target_angles(model, data, ball_position, target_position, initialize_angles, body_id, jacp, jacr, movable_joints_indices, task_name)
    
    if task_name[0] == "move" and task_name[1] == "lower above" and task_name[2] == "ball position":
        print("moving to the lower above of ball position")
        FLAG = move(steps, niryo, target_angles, FLAG, joint_indices, data, site_id, body_id)
    elif task_name[0] == "move" and task_name[1] == "higher above" and task_name[2] == "ball position":
        print("lift the ball to higher above position")
        FLAG = move(steps, niryo, target_angles, FLAG, joint_indices, data, site_id, body_id)
    elif task_name[0] == "move" and task_name[1] == "exact" and task_name[2] == "ball position":
        print("moving down to the ball position")
        FLAG = move(steps, niryo, target_angles, FLAG, joint_indices, data, site_id, body_id)


    elif task_name[0] == "grab" and task_name[1] == "ball":
        print("close the gripper\n")
        FLAG = close_the_gripper(steps, niryo, target_angles, FLAG, joint_indices, data, site_id, body_id)
    elif task_name[0] == "release" and task_name[1] == "ball":
        print("release the gripper")
        FLAG = open_the_gripper(steps, niryo, target_angles, FLAG, joint_indices, data, site_id, body_id)

    elif task_name[0] == "move" and task_name[1] == "lower above" and task_name[2] == "target position":
        print("move down to the lower above of target position")
        FLAG = move(steps, niryo, target_angles, FLAG, joint_indices, data, site_id, body_id)
    elif task_name[0] == "move" and task_name[1] == "exact" and task_name[2] == "target position":
        print("move to the target position")
        FLAG = move(steps, niryo, target_angles, FLAG, joint_indices, data, site_id, body_id)
    elif task_name[0] == "move" and task_name[1] == "higher above" and task_name[2] == "target position":
        print("move to the higher above of target position")
        FLAG = move(steps, niryo, target_angles, FLAG, joint_indices, data, site_id, body_id)
    
    elif task_name[0] == "return" and task_name[1] == "initial position":
        print("move to the initial position")
        FLAG = move(steps, niryo, target_angles, FLAG, joint_indices, data, site_id, body_id)
    
    return FLAG

# Simulation loop for synchronized movement
with viewer.launch_passive(model, data) as Viewer:
    
    Viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 0
    Viewer.sync()
    ball_position = get_ball_position(data, ball_body_id)
    print("Initial ball position:", ball_position)
    # 0.23, -0.16, 0.107
    
    target_position = np.array([0.23, 0.16, 0.107])
    print("Target position:", target_position)
    
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
    duration = 10  # seconds
    steps_1 = int(10 * 50)
    steps_2 = int(10 * 50)
    steps_3 = int(10 * 50)
    steps_4 = int(10 * 50)
    steps_5 = int(10 * 50)
    steps_6 = int(10 * 50)
    Steps = int(10 * 50)

    # Set initial joint positions
    for name, angle in fixed_positions.items():
        joint_index = joint_indices[name]
        data.qpos[joint_index] = angle

    Viewer.sync()
    time.sleep(0.02)
    
    angle_list = []
    
    print("calculating angle values...")
    for task_name in task_list:
        target_angles = get_task_name_and_target_angles(model, data, ball_position, target_position, initialize_angles, body_id, jacp, jacr, movable_joints_indices, task_name)
        print("Target angles:", target_angles)
        angle_list.append(target_angles)

    print("Starting simulation...")

    for task_name in task_list:
        target_angles = angle_list.pop(0)
        FLAG = control_command(Steps, niryo, target_angles, FLAG, joint_indices, data, site_id, body_id, task_name)
        Viewer.sync()
        time.sleep(1)
    
    print("Simulation finished.")
