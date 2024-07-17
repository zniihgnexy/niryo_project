from ikpy.chain import Chain
from ikpy.link import URDFLink, OriginLink
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# loadlink from urdf file
urdf_file_path = "niryo_arm.xml"
robot_chain_2 = Chain.from_urdf_file(urdf_file_path)

target_position_2 = [0.2, 0.2, 0.1]  # Example target position

# breakpoint()

angles_second = robot_chain_2.inverse_kinematics(target_position_2)

# print("chain names:", robot_chain_2.name)
print("Inverse Kinematics Joint Angles 2:", angles_second)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
robot_chain_2.plot(angles_second, ax)
plt.show()

position_testify = robot_chain_2.forward_kinematics(angles_second)
print("Position for the angles: ", position_testify)