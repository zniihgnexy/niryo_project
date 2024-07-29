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

# Simulation loop for synchronized movement
with viewer.launch_passive(model, data) as Viewer:
    niryo = RobotController(model, data, joint_names, pids, joint_indices=joint_indices)
    Viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 0
    Viewer.sync()
    ball_position = get_ball_position(data, ball_body_id)
    print("Initial ball position:", ball_position)
    # 0.23, -0.16, 0.107
    
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
    # Pre-calculate target angles for each step
    task_name_1 = "move to lower above of ball position"
    target_position_1 = ball_position + np.array([0, 0, 0.1])
    target_angles_1 = pre_calc(model, data, target_position_1, initialize_angles, body_id, jacp, jacr, movable_joints_indices)
    
    task_name_2 = "move to ball position"
    target_position_2 = ball_position + np.array([0, 0, 0.04])
    target_angles_2 = pre_calc(model, data, target_position_2, initialize_angles, body_id, jacp, jacr, movable_joints_indices)
    
    task_name_3 = "close the gripper"
    target_position_3 = ball_position + np.array([0, 0, 0.04])
    target_angles_3 = pre_calc(model, data, target_position_3, initialize_angles, body_id, jacp, jacr, movable_joints_indices)
    
    task_name_4 = "move to higher above of ball position"
    target_position_4 = ball_position + np.array([0, 0, 0.15])
    target_angles_4 = pre_calc(model, data, target_position_4, initialize_angles, body_id, jacp, jacr, movable_joints_indices)
    
    target_position = np.array([0.23, 0.16, 0.107])
    task_name_5 = "move to higher above of target position"
    target_position_5 = target_position + np.array([0, 0, 0.15])
    target_angles_5 = pre_calc(model, data, target_position_5, initialize_angles, body_id, jacp, jacr, movable_joints_indices)
    
    task_name_6 = "move to lower above of target position"
    target_position_6 = target_position + np.array([0, 0, 0.1])
    target_angles_6 = pre_calc(model, data, target_position_6, initialize_angles, body_id, jacp, jacr, movable_joints_indices)
    
    task_name_7 = "move to target position"
    target_position_7 = target_position + np.array([0, 0, 0.04])
    target_angles_7 = pre_calc(model, data, target_position_7, initialize_angles, body_id, jacp, jacr, movable_joints_indices)
    
    task_name_8 = "release the gripper"
    target_position_8 = target_position + np.array([0, 0, 0.04])
    target_angles_8 = pre_calc(model, data, target_position_8, initialize_angles, body_id, jacp, jacr, movable_joints_indices)
    
    task_name_9 = "move to higher above of target position"
    target_position_9 = target_position + np.array([0, 0, 0.15])
    target_angles_9 = pre_calc(model, data, target_position_9, initialize_angles, body_id, jacp, jacr, movable_joints_indices)
    
    # breakpoint()
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

    print("Starting simulation...")
    print("Step 1: Move towards the ball")
    for step in range(steps_1):
        target_angles_1 = check.updates_gripper_angles(niryo, target_angles_1, FLAG, joint_indices)
        RobotController.control_arm_to_position(niryo, target_angles_1)
        # end_effector_pos = data.site(site_id).xpos
        RobotController.sync_viewer(niryo, Viewer)
        time.sleep(0.02)
    
    # breakpoint()
    
    print("Step 2: moving down")
    for step in range(steps_2):
        target_angles_2 = check.updates_gripper_angles(niryo, target_angles_2, FLAG, joint_indices)
        RobotController.control_arm_to_position(niryo, target_angles_2)
        
        # if fit in the middle of the gripper then return True and FLAG to 1
        if check.check_before_close_gripper(data, body_id, target_position, 0.01):
            FLAG = 1

        RobotController.sync_viewer(niryo, Viewer)
        time.sleep(0.02)
    FLAG = 1
    print("Step 3: close the gripper\n")
    for step in range(steps_3):
        # close gripper angles
        target_angles_3 = check.updates_gripper_angles(niryo, target_angles_3, FLAG, joint_indices)
        RobotController.control_arm_to_position(niryo, target_angles_3)
                
        if FLAG == 1:
            ball_pos = check.ball_pos_update(data, site_id)
            RobotController.set_mocap_position(niryo, "box", ball_pos)
            Viewer.sync()
        
        RobotController.sync_viewer(niryo, Viewer)
        time.sleep(0.02)

    print("Step 4: lift the ball to higher above position")
    for step in range(steps_4):
        target_angles_4 = check.updates_gripper_angles(niryo, target_angles_4, FLAG, joint_indices)
        RobotController.control_arm_to_position(niryo, target_angles_4)
        
        if FLAG == 1:
            ball_pos = check.ball_pos_update(data, site_id)
            RobotController.set_mocap_position(niryo, "box", ball_pos)
            Viewer.sync()
        
        RobotController.sync_viewer(niryo, Viewer)
        time.sleep(0.02)
        
    print("Step 5: move to the higher above of target position")
    for step in range(steps_5):
        target_angles_5 = check.updates_gripper_angles(niryo, target_angles_5, FLAG, joint_indices)
        RobotController.control_arm_to_position(niryo, target_angles_5)
        
        if FLAG == 1:
            ball_pos = check.ball_pos_update(data, site_id)
            RobotController.set_mocap_position(niryo, "box", ball_pos)
            Viewer.sync()
        
        RobotController.sync_viewer(niryo, Viewer)
        time.sleep(0.02)
    
    print("Step 6: move down to the lower above of target position")
    for step in range(steps_6):
        target_angles_6 = check.updates_gripper_angles(niryo, target_angles_6, FLAG, joint_indices)
        RobotController.control_arm_to_position(niryo, target_angles_6)

        if FLAG == 1:
            ball_pos = check.ball_pos_update(data, site_id)
            RobotController.set_mocap_position(niryo, "box", ball_pos)
            Viewer.sync()
        
        RobotController.sync_viewer(niryo, Viewer)
        time.sleep(0.02)
        
    print("Step 7: move to the target position")
    for step in range(Steps):
        target_angles_7 = check.updates_gripper_angles(niryo, target_angles_7, FLAG, joint_indices)
        RobotController.control_arm_to_position(niryo, target_angles_7)
        
        if FLAG == 1:
            ball_pos = check.ball_pos_update(data, site_id)
            RobotController.set_mocap_position(niryo, "box", ball_pos)
            Viewer.sync()
        
        RobotController.sync_viewer(niryo, Viewer)
        time.sleep(0.02)
    
    FLAG = 0
    
    print("Step 8: release the gripper")
    for step in range(Steps):
        target_angles_8 = check.updates_gripper_angles(niryo, target_angles_8, FLAG, joint_indices)
        RobotController.control_arm_to_position(niryo, target_angles_8)
        
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
    
    print("Step 9: move to the higher above of target position")
    for step in range(Steps):
        target_angles_9 = check.updates_gripper_angles(niryo, target_angles_9, FLAG, joint_indices)
        RobotController.control_arm_to_position(niryo, target_angles_9)
        
        RobotController.sync_viewer(niryo, Viewer)
        time.sleep(0.02)

    print("All joints have moved towards their target positions.")
    # print("End-effector final position:", end_effector_pos)
