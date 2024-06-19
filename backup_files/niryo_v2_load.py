import mujoco
import numpy as np
import time
import mujoco.viewer

# Load your model
model = mujoco.MjModel.from_xml_path('/home/xz2723/niryo_project/meshes/niryo_arm_table.xml')
data = mujoco.MjData(model)

# Create a viewer for the simulation
viewer = mujoco.viewer.launch(model, data)
# Set initial joint angles
initial_angles = [0, 0.5, -0.5, 0, 0, 0]
# sim_state = model.qpos
# set the initial state value as sim_MjData
sim_MjData = mujoco.MjData(model)
for i in range(len(initial_angles)):
    sim_MjData.qpos[i] = initial_angles[i]
    print("this state of this angle", sim_MjData.qpos[i])

print("this is the initial state", sim_MjData.qpos)


import pdb; pdb.set_trace()

mujoco.mj_forward(model, sim_MjData)
print("this is the new model", model)
updated_data = mujoco.MjData(model)
print("this is the state", updated_data.qpos)

# Function to set a specific joint angle
def set_joint_angle(model, data, joint_name, angle):
    print("et the joint angle of joint %s to %f" % (joint_name, angle))
    joint_id = model.joint_name2id(joint_name)
    data.qpos[joint_id] = angle
    mujoco.mj_forward(model, data)

print("Starting robot arm control simulation...")

# Set initial angles for all joints to zero for simplicity
initial_angles = np.zeros(model.nq)
data.qpos[:len(initial_angles)] = initial_angles
mujoco.mj_forward(model, data)
