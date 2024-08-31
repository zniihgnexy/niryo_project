import mujoco
"""
This script calculates and visualizes the positions and lengths of joints in a robot arm.

Functions:
- get_joint_positions(data): Returns the 3D positions of each joint in the robot arm.
"""

import numpy as np
import time
from mujoco import viewer
import matplotlib.pyplot as plt
from pid_controller import PIDController
from kinemetic import robot_chain



# Load the model
model_path = '/home/xz2723/niryo_project/meshes/mjmodel.xml'
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Joint names and initialization
joint_names = ['world', 'base_link', 'shoulder_link', 'arm_link', 'elbow_link', 'forearm_link', 'wrist_link', 'hand_link', 'g1_mainSupport', 'g1_clampLeft', 'g1_clampRight']
# joint_indices is the index number of joint_names
joint_indices = {name: joint_names.index(name) for name in joint_names}
print("joint_indices: ", joint_indices)

# Set all joint positions to zero
for name in joint_names:
    joint_index = joint_indices[name]
    data.qpos[joint_index] = 0.0

# Update the model to apply the zero positions
mujoco.mj_forward(model, data)

# breakpoint()
# Function to get the 3D positions of each joint
def get_joint_positions(data):
    positions = []
    for i in range(len(joint_names)):
        position = model.body(joint_indices[joint_names[i]]).pos
        # geom_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, joint_names[i])
        print("the name of the joint is: ", joint_names[i])
        # print(" the id of the joint is: ", geom_index)
        positions.append(position)
        print("current position is: ", position)
    return np.array(positions)

# Get the positions of each joint
joint_positions = get_joint_positions(data)

# Calculate the lengths between consecutive joints
joint_lengths = np.linalg.norm(np.diff(joint_positions, axis=0), axis=1)
print("Joint lengths:", joint_lengths)

# Plot the joint positions to visualize the arm structure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(joint_positions[:, 0], joint_positions[:, 1], joint_positions[:, 2], 'bo-', linewidth=2)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

# Print the final positions and lengths of each joint
print("Joint positions:\n", joint_positions)
print("Lengths between joints:\n", joint_lengths)
