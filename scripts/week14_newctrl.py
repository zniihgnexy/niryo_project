import math
import mujoco
import numpy as np
import time
from mujoco import viewer
import matplotlib.pyplot as plt
from pid_controller import PIDController, PIDControllerWithDerivativeFilter
from GradientDescentIK import GradientDescentIK
from LevenbergMarquardtIK import LevenbergMarquardtIK

# Load the model
model_path = '/home/xz2723/niryo_project/meshes/mjmodel.xml'
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Get the ID of the ball body
ball_body_id = model.body('box').id

# Function to get the position of the ball
def get_ball_position(data, body_id):
    return data.body(body_id).xpos

# ball_position = get_ball_position(data, ball_body_id)
# print("Initial box position:", ball_position)

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
body_id = model.body('gripper_clamp_left').id
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

def get_target_angles(model, data, target_position, initialize_angles, body_id, jacp, jacr, movable_joints_indices):
    ik = GradientDescentIK(model, data, step_size, tol, alpha, jacp, jacr, movable_joints_indices)
    ik_lm = LevenbergMarquardtIK(model, data, step_size, tol, alpha, 0.1, movable_joints_indices)    
    target_angles = ik.calculate(target_position, initialize_angles[:8], body_id)
    print("Target angles:", target_angles)
    return target_angles

def soft_start_control(current_position, target_position, step_size, max_increment):
    # Calculate incremental step
    direction = np.sign(target_position - current_position)
    increment = direction * min(step_size, abs(target_position - current_position), max_increment)
    return current_position + increment

def compute_feedforward(target_position, dynamic_parameters):
    # Placeholder for a dynamic model or empirical data
    # For simplicity, just return a fraction of the target position
    return 0.1 * target_position

def control_with_feedforward(pid, target_position, current_position):
    feedforward_value = compute_feedforward(target_position, {})
    pid_output = pid.calculate(target_position - current_position)
    return current_position + pid_output + feedforward_value


# breakpoint()

# Initialize PID controllers
pids = {
    'joint_1': PIDController(100, 0.5, 100),
    'joint_2': PIDController(100, 0.8, 100),
    'joint_3': PIDController(100, 0.06, 100),
    'joint_4': PIDController(180, 0.06, 100),
    'joint_5': PIDController(100, 0.0001, 150),
    'joint_6': PIDControllerWithDerivativeFilter(162.5, 0.06, 100),
    'left_clamp_joint': PIDControllerWithDerivativeFilter(10, 0.005, 50),
    'right_clamp_joint': PIDControllerWithDerivativeFilter(10, 0.005, 50)
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

def control_arm_to_position_nogripper(model, data, joint_names, joint_indices, pids, target_angles, step_size=0.001, max_increment=0.0005):
    for name in joint_names:
        joint_index = joint_indices[name]
        current_position = data.qpos[joint_index]
        
        if name in ['left_clamp_joint', 'right_clamp_joint']:
            target_position = -0.0051 if name == 'left_clamp_joint' else 0.00515
            new_position = soft_start_control(current_position, target_position, step_size, max_increment)
            control_signal = pids[name].calculate(new_position, current_position)
            # keep the exact same position as before
            # data.ctrl[joint_index] = 0
            # continue
        else:
            target_position = target_angles[joint_indices[name]]
            new_position = soft_start_control(current_position, target_position, step_size, max_increment)
            control_signal = pids[name].calculate(target_position, current_position)

        # control_signal = pids[name].calculate(new_position, current_position)
        data.ctrl[joint_index] = control_signal
        joint_angle_history[name].append(current_position)
    
    position_updates.append(data.site(site_id).xpos)

def control_arm(data, joint_indices, target_angles):
    """Control only the arm joints, keeping them stable."""
    for name in ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']:
        joint_index = joint_indices[name]
        current_position = data.qpos[joint_index]
        target_angle = target_angles[joint_indices[name]]
        pid = pids[name]
        control_signal = pid.calculate(target_angle, current_position)
        data.ctrl[joint_index] = control_signal
        joint_angle_history[name].append(current_position)

def control_grippers(data, joint_indices, close=True, step_size=0.00005):
    """Incrementally close or open the grippers and check for secure gripping."""
    target_position_left = -0.012 if close else 0
    target_position_right = 0.012 if close else 0
    left_index = joint_indices['left_clamp_joint']
    right_index = joint_indices['right_clamp_joint']
    current_position_left = data.qpos[left_index]
    current_position_right = data.qpos[right_index]

    # Moving towards the target positions incrementally
    while (close and (current_position_left > target_position_left or current_position_right < target_position_right)) or \
          (not close and (current_position_left < 0 or current_position_right > 0)):
        if close:
            current_position_left -= step_size
            current_position_right += step_size
            current_position_left = max(current_position_left, target_position_left)
            current_position_right = min(current_position_right, target_position_right)
        else:
            current_position_left += step_size
            current_position_right -= step_size
            current_position_left = min(current_position_left, 0)
            current_position_right = max(current_position_right, 0)
        
        data.qpos[left_index] = current_position_left
        data.qpos[right_index] = current_position_right
        mujoco.mj_step(data.model, data)  # Update the simulation to reflect new positions

        # Check for contact between gripper and box
        if has_contact(data, 'g1_clampLeft', 'g1_clampRight', 'box'):
            print("Gripper has successfully grabbed the box.")
            return True

    return False

def has_contact(data, gripper_left_id, gripper_right_id, box_id):
    """Check specific contact between grippers and box."""
    for i in range(data.ncon):  # Check all current contacts
        contact = data.contact[i]
        if contact.geom1 in (gripper_left_id, gripper_right_id) and contact.geom2 == box_id or \
            contact.geom2 in (gripper_left_id, gripper_right_id) and contact.geom1 == box_id:
            return True
    return False

# Improved stage 3 to lift the ball
def lift_to_new_position(model, data, initial_position, lift_height, joint_indices, body_id, steps):
    target_position = initial_position + np.array([0, 0, lift_height])
    # breakpoint()
    target_angles = get_target_angles(model, data, target_position, initialize_angles, body_id, jacp, jacr, movable_joints_indices)
    mujoco.mj_step(model, data)
    Viewer.sync()
    return target_angles

def set_mocap_position(model, data, body_name, position):
    """Set the position of a mocap body."""
    body_idx = model.body(name=body_name).id
    mocap_idx = model.body_mocapid[body_idx]
    if mocap_idx == -1:
        raise ValueError("Body is not a mocap body or does not exist.")
    
    # Set the position
    data.mocap_pos[mocap_idx][:] = np.array(position)

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
    
    target_position = ball_position + np.array([0, 0, 0.015])
    target_angles_2ball = get_target_angles(model, data, target_position, initialize_angles, body_id, jacp, jacr, movable_joints_indices)
    for name, angle in fixed_positions.items():
        # change the values of joint 6
        if name == 'joint_6':
            fixed_positions[name] = -0.10000
    print("target_angles_2ball:", target_angles_2ball)
    
    target_position = target_position + np.array([0, 0, 0.003])
    target_close_gripper = get_target_angles(model, data, target_position, initialize_angles, body_id, jacp, jacr, movable_joints_indices)
    
    list_position = [0.23, -0.15, 0.30]
    target_angles_lift = get_target_angles(model, data, list_position, initialize_angles, body_id, jacp, jacr, movable_joints_indices)
    print("target_angles_oripos:", target_angles_lift)
    
    # breakpoint()
    # Simulation parameters
    duration = 10  # seconds
    steps_1 = int(10 * 50)
    steps_2 = int(10 * 50)
    steps_3 = int(10 * 50)
    steps_4 = int(10 * 50)

    # Set initial joint positions
    for name, angle in fixed_positions.items():
        joint_index = joint_indices[name]
        data.qpos[joint_index] = angle

    # breakpoint()
    # mujoco.mj_step(model, data)
    # test the ball's mocap
    # test_mocap_pos = [0.1, 0.2, 0.3]
    # set_mocap_position(model, data, "box", test_mocap_pos)
    # mujoco.mj_step(model, data)
    
    Viewer.sync()
    time.sleep(0.02)

    print("Starting simulation...")
    print("Step 1: Move towards the ball")
    for step in range(steps_1):
        # Update control signals for all joints at each step
        control_arm_to_position(model, data, joint_names, joint_indices, pids, target_angles_onball)
        
        end_effector_pos = data.body(body_id).xpos
        # position_updates.append(end_effector_pos)
        
        mujoco.mj_step(model, data)
        Viewer.sync()
        time.sleep(0.02)
        
    
    print("Step 2: moving down")
    for step in range(steps_2):
        # Update control signals for all joints at each step
        control_arm_to_position(model, data, joint_names, joint_indices, pids, target_angles_2ball)
        
        end_effector_pos = data.body(body_id).xpos
        
        error_tolerance = math.sqrt((end_effector_pos[0] - target_position[0])**2 + (end_effector_pos[1] - target_position[1])**2 + (end_effector_pos[2] - target_position[2])**2)
        
        if FLAG == 0 and error_tolerance < 0.02:
            FLAG = 1
            # control_gripper(data, joint_indices, close=True)
            print("Ball is picked up")

        mujoco.mj_step(model, data)
        Viewer.sync()
        time.sleep(0.02)
    
    print("Step 3: close the gripper\n")
    for step in range(steps_3):
        # Call the modified function with a small step size for a gradual close
        control_arm_to_position_nogripper(model, data, joint_names, joint_indices, pids, target_angles_2ball, step_size=0.01)
        # control_only_grippers(data, joint_indices, close=True)
        
        # see if the grabbing is secured
        if FLAG == 1:
            mocap_pos = end_effector_pos
            set_mocap_position(model, data, "box", mocap_pos)
        
        breakpoint()
        
        mujoco.mj_step(model, data)
        Viewer.sync()
        time.sleep(0.02)

    print("Step 4: lift the ball")
    for step in range(steps_4):
        control_arm_to_position_nogripper(model, data, joint_names, joint_indices, pids, target_angles_lift)
        
        end_effector_pos = data.body(body_id).xpos
        
        if FLAG == 1:
            mocap_pos = end_effector_pos
            set_mocap_position(model, data, "box", mocap_pos)
        
        mujoco.mj_step(model, data)
        Viewer.sync()
        time.sleep(0.02)

    print("All joints have moved towards their target positions.")
    print("End-effector final position:", end_effector_pos)

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
