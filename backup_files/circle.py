import mujoco
import numpy as np
import time
from mujoco import viewer

# Load the model
model_path = '/home/xz2723/niryo_project/meshes/mjmodel.xml'
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Define the circle function
def circle(t, radius, frequency, center=(0, 0, 0)):
    """ Calculate the position on a circle. """
    x = center[0] + radius * np.cos(2 * np.pi * frequency * t)
    y = center[1] + radius * np.sin(2 * np.pi * frequency * t)
    return np.array([x, y, center[2]])

# Simulation parameters
frequency = 0.1  # Frequency of the circle
radius = 0.1     # Radius of the circle
center = [0.0, 0.0, 0.15]  # Center position of the circle in 3D space
duration = 10  # Total duration of the simulation in seconds
num_steps = int(duration / model.opt.timestep)  # Total number of simulation steps

# Use viewer.launch to open the visualization
with viewer.launch(model, data) as viewer:
    start_time = time.time()
    for i in range(num_steps):
        # Calculate the current time within the simulation
        sim_time = time.time() - start_time

        # Update the target position based on the circle function
        target_pos = circle(sim_time, radius, frequency, center)

        # Simple PD control to move the gripper center to the target position
        position_error = target_pos - data.site_xpos[0]  # Assuming 'gripper_center' is the first site
        velocity_command = 0.1 * position_error  # Proportional control

        # Apply velocity command to the joints
        for index, joint_id in enumerate(data.actuator_trnid[:, 0]):
            data.ctrl[index] = velocity_command[joint_id] * 0.1  # Scale velocity command

        # Step the simulation
        mujoco.mj_step(model, data)

        # Update the viewer
        viewer.render()
