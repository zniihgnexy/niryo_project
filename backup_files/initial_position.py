import mujoco
import numpy as np
import time

# Load the pendulum model
model = mujoco.MjModel.from_xml_path('/path/to/mujoco/models/pendulum.xml')
data = mujoco.MjData(model)

# Initialize the viewer
viewer = mujoco.MjViewer(model, data)

# Run the simulation forward to reflect initial conditions
mujoco.mj_forward(model, data)

# Simulation parameters
duration = 5  # seconds
time_step = 0.02  # seconds
num_steps = int(duration / time_step)

# Define a simple sinusoidal motion for the pendulum joint
time_array = np.linspace(0, duration, num_steps)
target_positions = np.pi / 2 * np.sin(2 * np.pi * 0.5 * time_array)  # 0.5 Hz sine wave

# Simulation loop
for i in range(num_steps):
    # Set the joint position for the current step
    data.qpos[0] = target_positions[i]

    # Run one step of the simulation
    mujoco.mj_step(model, data)

    # Update the viewer
    viewer.sync(data)

    # Delay to match the desired time step
    time.sleep(time_step)

print("Simulation ended.")
