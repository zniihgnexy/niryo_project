import mujoco
import numpy as np
import time
import mujoco.viewer

model = mujoco.MjModel.from_xml_path('/home/xz2723/niryo_project/meshes/niryo_arm_table.xml')
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)

initial_angles = np.array([1, 0.5, -0.5, 0, 0, 0])

data.qpos[:len(initial_angles)] = initial_angles
mujoco.mj_forward(model, data)  # Compute the forward dynamics

viewer = mujoco.viewer.launch(model, data)

print("Starting robot arm control simulation...")

# Simulation loop
while not mujoco.viewer is None:
    data.qpos[1] -= 0.01
    
    if data.qpos[1] < 0:
        data.qpos[1] = 0.5  # Reset to initial value if it exceeds 1.0
    
    mujoco.mj_step(model, data)
    renderer.render(data)

    time.sleep(0.01)

print("Simulation ended.")
