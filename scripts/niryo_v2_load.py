import mujoco
import numpy as np
import time
import mujoco.viewer

# Load your model
# model = mujoco.MjModel.from_xml_path('/home/xz2723/niryo_project/meshes/niryo_robot.urdf')
model = mujoco.MjModel.from_xml_path('/home/xz2723/niryo_project/meshes/mjmodel.xml')
data = mujoco.MjData(model)

initial_angles = [0, 0.5, -0.5, 0, 0, 0]

data.qpos[:len(initial_angles)] = initial_angles
mujoco.mj_forward(model, data)

# Create a viewer for the simulation
viewer = mujoco.viewer.launch(model, data)

# breakpoint()

# only get the position of the joints
# joint_names = model.joint_names

'''
position for mjmodle part:
the geom position 
[-0.22        0.2         0.11463309] 
[0.30412458 0.01858042 0.12479942] 
[ 0.30411079 -0.0189203   0.12478893] 
[0.2792963  0.00143164 0.14806996] 
[ 2.83796279e-01 -2.73314741e-05  1.62681903e-01] 
[ 2.68465084e-01 -1.57409831e-05  2.15191519e-01] 
[ 0.25456814 -0.00519176  0.26355595] 
[0.19131902 0.00425687 0.41715656] 
[ 0.09640099 -0.00301431  0.3505623 ] 
[-0.01472569  0.00505786  0.24716807] 
[0.   0.   0.05] 
[-0.00798452 -0.00038603  0.14974837]

position for urdf part:
the geom position 
[ 0.17038349  0.00625954 -0.09421746] 
[ 0.18614607 -0.01488094 -0.09832342] 
[ 0.18622355  0.00238186 -0.06680629] 
[ 0.18156302 -0.00819551 -0.0667679 ] 
[ 0.17962403 -0.01003134 -0.01878469] 
[ 0.20493406 -0.00699107  0.02899439] 
[0.23363213 0.00090081 0.19285023] 
[ 0.11755302 -0.00470345  0.18810544] 
[-0.01465153  0.00526885  0.14716807] 
[-0.00798452 -0.00038603  0.04974837]

'''

print("the geom position", data.geom(9).xpos, data.geom(8).xpos, data.geom(7).xpos, data.geom(6).xpos, data.geom(5).xpos, data.geom(4).xpos, data.geom(3).xpos, data.geom(2).xpos, data.geom(1).xpos, data.geom(0).xpos)

# Set initial joint angles
initial_angles = [0, 0.5, -0.5, 0, 0, 0]
# sim_state = model.qpos
# set the initial state value as sim_MjData
sim_MjData = mujoco.MjData(model)
for i in range(len(initial_angles)):
    sim_MjData.qpos[i] = initial_angles[i]
    print("this state of this angle", sim_MjData.qpos[i])

print("this is the initial state", sim_MjData.qpos)


# import pdb; pdb.set_trace()

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
