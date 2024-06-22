import mujoco
import numpy as np
import time
from mujoco import viewer
import matplotlib.pyplot as plt

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral_error = 0
        self.prev_error = 0
        self.dt = 0.02

    def calculate(self, target, current):
        error = target - current
        self.integral_error += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        output = (self.Kp * error) + (self.Ki * self.integral_error) + (self.Kd * derivative)
        self.prev_error = error
        return output


# Load the model
model_path = '/home/xz2723/niryo_project/meshes/mjmodel.xml'
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Initialize angles
initialize_angles = np.array([1.0000, 0.5000, 0.5000, 0.0000, 0.0000, 0.0000, 0.00000, 0.00000])

# Joint names and target angles
joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'left_clamp_joint', 'right_clamp_joint']
target_angles = {
    'joint_1': 0.1000, 
    'joint_2': 0.2000, 
    'joint_3': 0.3000, 
    'joint_4': 0.1000, 
    'joint_5': 0.2000, 
    'joint_6': 0.3000, 
    'left_clamp_joint': 0.00000, 
    'right_clamp_joint': 0.00000
}

# 0.1, 0.2, 0.3, 0.1, 0.2, 0.1

# get the joint values from the target_angles dictionary
target_angles_values = np.array([target_angles[name] for name in joint_names])

# Map joint names to indices using mj_name2id
joint_indices = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names}

joint_angle_history = {name: [] for name in joint_names}

# Initialize joint positions according to target angles
for name in joint_names:
    joint_id = joint_indices[name]
    data.qpos[joint_id] = initialize_angles[joint_indices[name]]
    joint_angle_history[name].append(data.qpos[joint_id])

# get the 3D position
def get_3d_position(data, joint_names):
    joint_pos = {}
    for name in joint_names:
        joint_id = joint_indices[name]
        joint_pos[name] = data.geom(joint_id).xpos
    return joint_pos

with viewer.launch_passive(model, data) as Viewer:
    # Set viewer settings, such as enabling wireframe mode
    Viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 1
    Viewer.sync()

    # Hold initial position
    initial_hold_time = 1  # seconds
    start_time = time.time()
    print("Holding initial position for 2 seconds...")
    while time.time() - start_time < initial_hold_time:
        Viewer.sync()
        time.sleep(0.05)
    print("Initial position hold complete.")
    # breakpoint()
    
    position_of_joints = get_3d_position(data, joint_names)
    print(f"Initial position of each joint: {position_of_joints}")
    
    print("the position of geom 6: ", data.geom(6).xpos)
    
    # angles = my_chain.inverse_kinematics(target_vector, target_orientation)