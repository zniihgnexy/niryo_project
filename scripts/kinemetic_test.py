from ikpy.chain import Chain
from ikpy.link import URDFLink
import numpy as np

# from roboticstoolbox import Chain, URDFLink

robot_chain = Chain(name='niryo_one', links=[
    URDFLink(
        name="base",
        origin_translation=[0, 0, 0.1],
        origin_orientation=[0, 0, 0],
        rotation=[0, 0, 0],
        bounds=None
    ),
    URDFLink(
        name="joint_1",
        origin_translation=[0, 0, 0.103],
        origin_orientation=[0, 0, 0],
        rotation=[0, 0, 1],
        bounds=(-2.949, 2.949)
    ),
    URDFLink(
        name="joint_2",
        origin_translation=[0, 0, 0.08],
        origin_orientation=[0.0, 1.5707963267948966, 0.0],  
        rotation=[0, 0, 1],
        bounds=(-1.83, 0.61)
    ),
    URDFLink(
        name="joint_3",
        origin_translation=[0.21, 0, 0],
        origin_orientation=[-1.5707969456925137, -0.0, 0.0],  # Adjust orientation if necessary
        rotation=[0, 0, 1],
        bounds=(-1.34, 1.57)
    ),
    URDFLink(
        name="joint_4",
        origin_translation=[0.0415, 0.03, 0],
        origin_orientation=[3.141592653589793, -1.5707963267948966, 3.141592653589793],  # Adjust orientation if necessary
        rotation=[0, 0, 1],
        bounds=(-2.089, 2.089)
    ),
    URDFLink(
        name="joint_5",
        origin_translation=[0, 0, 0.19],
        origin_orientation=[3.141592653589793, 1.5707963267948966, 3.141592653589793],  # Adjust orientation if necessary
        rotation=[0, 0, 1],
        bounds=(-1.74533, 1.91986)
    ),
    URDFLink(
        name="joint_6",
        origin_translation=[0.0164, -0.0055, 0],
        origin_orientation=[3.141592653589793, -1.5707963267948966, 3.141592653589793],  # Adjust orientation if necessary
        rotation=[0, 0, 1],
        bounds=(-2.57436, 2.57436)
    ),
    URDFLink(
        name="left_clamp_joint",
        origin_translation=[0.027, 0, 0.015],
        origin_orientation=[1.5707923267988968, 0.0, -1.5707963267988967],  # Fixed joint
        rotation=[1, 0, 0],  # Slide along x-axis
        bounds=(-0.012, 0)
    ),
    URDFLink(
        name="right_clamp_joint",
        origin_translation=[0.027, 0, 0.015],
        origin_orientation=[1.5707923267988968, 0.0, -1.570796326798896],  # Fixed joint
        rotation=[1, 0, 0],  # Slide along x-axis
        bounds=(0, 0.012)
    )
])


# Example usage of the chain for IK
target_position = [0.1, 0.2, 0.1]  # Example target position
ik_angles = robot_chain.inverse_kinematics(target_position)

print("Inverse Kinematics Joint Angles 1:", ik_angles)

# # print the angle for joint two
# for i in range (0, len(ik_angles)):
#     print("Joint ", i+1, " angle: ", ik_angles[i])

# # Example of printing the links to verify their attributes
# for link in robot_chain.links:
#     print(link)

# ik_angles = robot_chain.inverse_kinematics([0.3, 0.3, 0.3])

# print("Inverse angles for the given position: ", ik_angles)

# # plot the robot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
robot_chain.plot(ik_angles, ax)    
plt.show()

# loadlink from urdf file
urdf_file_path = "niryo_robot.urdf"
robot_chain_2 = Chain.from_urdf_file(urdf_file_path)

# breakpoint()

angles_second = robot_chain_2.inverse_kinematics([0.1, 0.2, 0])

print("chain names:", robot_chain_2.name)
print("Inverse Kinematics Joint Angles 2:", angles_second)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
robot_chain_2.plot(angles_second, ax)
plt.show()

position_testify = robot_chain_2.forward_kinematics(angles_second)
print("Position for the angles: ", position_testify)