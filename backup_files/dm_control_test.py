import mujoco
import numpy as np
import time
from mujoco import viewer
from mujoco import _simulate

# Load the model
model_path = '/home/xz2723/niryo_project/meshes/Niryo_Ned2.robot'
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Initialize angles
initialize_angles = np.array([0.5, 0.5, 1, 0, 0, 0, 0, 0, 0.000, 0.000])

# Joint names and target angles
joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'left_clamp_joint', 'right_clamp_joint']
target_angles = {
    'joint_1': 1, 
    'joint_2': -0.5, 
    'joint_3': 1, 
    'joint_4': 0, 
    'joint_5': 0, 
    'joint_6': 0, 
    'left_clamp_joint': 0.000, 
    'right_clamp_joint': 0.000
}

# Give index to the names from the joint_names
joint_names_index = {name: idx for idx, name in enumerate(joint_names)}

# Initialize joint positions according to target angles
for name, angle in target_angles.items():
    joint_id = joint_names_index[name]
    data.qpos[joint_id] = initialize_angles[joint_id]


# Create a window to visualize the simulation
sim = _simulate.Simulate
print("Sim: ", sim)
print("sim type: ", type(sim))


Viewer = viewer.launch(model, data)

mujoco.mj_forward(model, data)

# Simulation parameters
duration = 10  # seconds
steps = int(duration * 50)  # Assuming 100 Hz simulation frequency
time_step = 1 / 50  # time step

# Simulation loop
for step in range(steps):
    # import pdb; pdb.set_trace()
    print("Step: ", step)
    for name, target_angle in target_angles.items():
        joint_id = joint_names_index[name]
        # Simple proportional controller for demonstration
        current_angle = data.qpos[joint_id]
        data.qpos[joint_id] += 0.05 * (target_angle - current_angle)
        print(f"Joint {name}: {current_angle:.2f} -> {data.qpos[joint_id]:.2f}")

    # Perform simulation step
    mujoco.mj_step(model, data)
        # renderer.render()  # Update the renderer
        # renderer.update_scene(data)  # Update the window

    # Sleep for the remainder of the time step
    time.sleep(time_step)

print("Simulation ended.")
