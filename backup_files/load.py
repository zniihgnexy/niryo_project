# import os
import mujoco
import numpy as np
import glfw
import mujoco.viewer

model = mujoco.MjModel.from_xml_path('/home/xz2723/mujoco-ur5-model/model/model_2.xml')
data = mujoco.MjData(model)

renderer = mujoco.Renderer(model)
# media = mujoco.Media(model)

mujoco.viewer.launch(model, data)
print('Total number of DoFs in the model:', model.nv)
print('Generalized positions:', data.qpos)
print('Generalized velocities:', data.qvel)

print('Joint names:', model.names)
print('data geom', data.geom_xpos)

mujoco.mj_forward(model, data)
renderer.update_scene(data)
print(model.geom('tableTop'))

# Load the XML model file
model_path = 'path/to/your/mujoco/model/ridgeback_ur5_robotiq_two_finger_gripper.xml'  # Change this to your actual file path
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Simulation loop
while True:
    # Apply control to the joint, for example, the shoulder_pan_joint
    shoulder_pan_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'shoulder_pan_joint')
    mujoco.mj_set_control(model, data, shoulder_pan_joint_id, 1.0)  # Set the control input, e.g., torque
    
    # Step the simulation
    mujoco.mj_step(model, data)
    
    # Optionally, retrieve the new joint position
    new_position = data.qpos[shoulder_pan_joint_id]

    #niryo ned2 - model
    
    print("New position of the shoulder_pan_joint:", new_position)
    
    # Break the loop or implement some stopping criteria
    # Here we just loop forever for demonstration purposes
    break  # Remove or modify this according to your loop condition

