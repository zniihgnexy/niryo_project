import math
import mujoco
import numpy as np
import time
from mujoco import viewer
import matplotlib.pyplot as plt
from pid_controller import PIDController
from GradientDescentIK import GradientDescentIK
from LevenbergMarquardtIK import LevenbergMarquardtIK

# Load the model
model_path = '/home/xz2723/niryo_project/meshes/mjmodel.xml'
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Get the ID of the ball body
ball_body_id = model.body('ball').id

# Function to get the position of the ball
def get_ball_position(data, body_id):
    return data.body(body_id).xpos

ball_position = get_ball_position(data, ball_body_id)
print("Initial ball position:", ball_position)

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
body_id = model.body('g1_mainSupport_link').id  # end-effector ID
site_id = model.site('gripper_center').id  # target site ID
jacp = np.zeros((3, model.nv))  # Translational Jacobian
jacr = np.zeros((3, model.nv))  # Rotational Jacobian
step_size = 0.01
tol = 0.001
alpha = 0.5
init_q = initialize_angles

movable_joints_indices = [joint_indices[joint_name] for joint_name in joint_names[:]]  # Exclude clamp joints

def get_target_angles(model, data, target_position, initialize_angles, body_id, jacp, jacr, movable_joints_indices):
    ik = GradientDescentIK(model, data, step_size, tol, alpha, jacp, jacr, movable_joints_indices)
    ik_lm = LevenbergMarquardtIK(model, data, step_size, tol, alpha, 0.1, movable_joints_indices)    
    target_angles = ik.calculate(target_position, initialize_angles[:8], body_id)
    print("Target angles:", target_angles)
    return target_angles

# breakpoint()

# Initialize PID controllers
pids = {
    'joint_1': PIDController(100, 0.5, 100),
    'joint_2': PIDController(100, 0.8, 100),
    'joint_3': PIDController(100, 0.06, 100),
    'joint_4': PIDController(180, 0.06, 100),
    'joint_5': PIDController(100, 0.0001, 150),
    'joint_6': PIDController(162.5, 0.06, 100),
    'left_clamp_joint': PIDController(10, 0.0001, 5),
    'right_clamp_joint': PIDController(10, 0.0001, 5)
}

joint_angle_history = {name: [] for name in joint_names}
position_updates = []

# Function to control arm position
def control_arm_to_position(model, data, joint_names, joint_indices, pids, target_angles):
    for name in joint_names:
        joint_index = joint_indices[name]
        current_position = data.qpos[joint_index]
        target_angle = target_angles[joint_indices[name]]
        control_signal = pids[name].calculate(target_angle, current_position)
        data.ctrl[joint_index] = control_signal
        joint_angle_history[name].append(current_position)
    
    position_updates.append(data.site(site_id).xpos)
    

FLAG = 0

# Simulation loop for synchronized movement
with viewer.launch_passive(model, data) as Viewer:
    Viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 0
    Viewer.sync()
    ball_position = get_ball_position(data, ball_body_id)
    print("Initial ball position:", ball_position)
    target_position = ball_position + np.array([0, 0, 0.05])
    print("target position:", target_position)
    target_angles_onball = get_target_angles(model, data, target_position, initialize_angles, body_id, jacp, jacr, movable_joints_indices)
    target_position = ball_position + np.array([0, 0, 0.01])
    target_angles_2ball = get_target_angles(model, data, target_position, initialize_angles, body_id, jacp, jacr, movable_joints_indices)
    
    # breakpoint()
    # Simulation parameters
    duration = 10  # seconds
    steps_1 = int(10 * 50)
    steps_2 = int(10 * 50)
    steps_3 = int(10 * 50)

    # Set initial joint positions
    for name, angle in fixed_positions.items():
        joint_index = joint_indices[name]
        data.qpos[joint_index] = angle

    Viewer.sync()
    time.sleep(0.02)

    print("Starting simulation...")
    print("Step 1: Move towards the ball")
    for step in range(steps_1):
        # Update control signals for all joints at each step
        control_arm_to_position(model, data, joint_names, joint_indices, pids, target_angles_onball)
        
        end_effector = data.body(body_id).xpos
        # position_updates.append(end_effector)
        
        mujoco.mj_step(model, data)
        Viewer.sync()
        time.sleep(0.02)  # Sleep to match the assumed simulation frequency
        
    
    print("Step 2: moving down")
    for step in range(steps_2):
        # Update control signals for all joints at each step
        control_arm_to_position(model, data, joint_names, joint_indices, pids, target_angles_2ball)
        
        end_effector = data.body(body_id).xpos
        # position_updates.append(end_effector)
        
        # breakpoint()
        
        error_tolerance = math.sqrt((end_effector[0] - target_position[0])**2 + (end_effector[1] - target_position[1])**2 + (end_effector[2] - target_position[2])**2)
        
        if FLAG == 0 and error_tolerance < 0.02:
            FLAG = 1
            print("Ball is picked up")

        mujoco.mj_step(model, data)
        Viewer.sync()
        time.sleep(0.02)
    
    print("Step 3: close the gripper\n")
    for step in range(steps_3):
        # Update control signals for all joints at each step
        data.ctrl[joint_indices['left_clamp_joint']] = 0.1
        data.ctrl[joint_indices['right_clamp_joint']] = 0.1
        
        end_effector = data.body(body_id).xpos
        # position_updates.append(end_effector)
        # update the ball position in the simulation
        data.body(ball_body_id).xpos = end_effector
        
        mujoco.mj_step(model, data)
        Viewer.sync()
        time.sleep(0.02)

    print("All joints have moved towards their target positions.")
    print("End-effector final position:", end_effector)
    

# make all the subpictures have teh same x and y axis scale
# Plot end-effector trajectory
position_updates = np.array(position_updates)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot(position_updates[:, 0], position_updates[:, 1], position_updates[:, 2])
ax.set_title('End-effector trajectory')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Plot joint angles over time
fig, axs = plt.subplots(4, 2, figsize=(15, 10))
for idx, name in enumerate(joint_names):
    ax = axs[idx // 2, idx % 2]
    ax.plot(np.linspace(0, duration, len(joint_angle_history[name])), joint_angle_history[name])
    ax.set_title(f'Trajectory of {name}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (rad)')

plt.tight_layout()
# plt.show()

# Print final joint angles
end_angles = [data.qpos[joint_indices[name]] for name in joint_names]
print("Final joint angles:", end_angles)
