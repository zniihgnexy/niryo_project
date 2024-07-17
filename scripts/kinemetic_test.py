from ikpy.chain import Chain
from ikpy.link import URDFLink, OriginLink
import numpy as np

# from roboticstoolbox import Chain, URDFLink

niryo_one_chain = Chain(name='niryo_one', links=[
    OriginLink(),
    URDFLink(
        name="shoulder_link",
        origin_translation=[0, 0, 0.203],
        origin_orientation=[0, 0, 0],
        rotation=[0, 0, 1]
    ),
    URDFLink(
        name="arm_link",
        origin_translation=[0, 0, 0.08],
        origin_orientation=[0.0, 1.5707963267948966, 0.0],  # Identity quaternion
        rotation=[0, 1, 0],
        bounds=[-2.949, 2.949]
    ),
    URDFLink(
        name="elbow_link",
        origin_translation=[0.21, 0, 0],
        origin_orientation=[-1.5707969456925137, -0.0, 0.0],  # Quaternion representation
        rotation=[0, 1, 0],
        bounds=[-1.83, 0.61]
    ),
    URDFLink(
        name="forearm_link",
        origin_translation=[0.0415, 0.03, 0],
        origin_orientation=[3.141592653589793, -1.5707963267948966, 3.141592653589793],  # Quaternion representation
        rotation=[0, 0, 1],
        bounds=[-1.34, 1.57]
    ),
    URDFLink(
        name="wrist_link",
        origin_translation=[0, 0, 0.19],
        origin_orientation=[3.141592653589793, 1.5707963267948966, 3.141592653589793],  # Quaternion representation
        rotation=[0, 1, 0],
        bounds=[-2.089, 2.089]
    ),
    URDFLink(
        name="hand_link",
        origin_translation=[0.0164, -0.0055, 0],
        origin_orientation=[3.141592653589793, -1.5707963267948966, 3.141592653589793],  # Quaternion representation
        rotation=[0, 0, 1],
        bounds=[-1.74533, 1.91986]
    ),
    URDFLink(
        name="g1_mainSupport_link",
        origin_translation=[0, 0, 0.0203],
        origin_orientation=[0.0, 1.546219296791616, 0.0],  # Quaternion representation
        rotation=[0, 0, 0],  # Slide joint
        bounds=[-0.012, 0.012]
    ),
    URDFLink(
        name="gripper_clamp_left",
        origin_translation=[0.007, 0, 0],
        origin_orientation=[1.5707923267988968, 0.0, -1.5707963267988967],  # Quaternion representation
        rotation=[1, 0, 0],  # Slide joint
        bounds=[-0.012, 0.012]
    ),
    URDFLink(
        name="gripper_clamp_right",
        origin_translation=[0.007, 0, 0],
        origin_orientation=[1.5707923267988968, 0.0, -1.5707963267988967],  # Quaternion representation
        rotation=[1, 0, 0],  # Slide joint
        bounds=[-0.012, 0.012]
    )
])


# Example usage of the chain for IK
target_position = [0.1, 0.2, 0.1]  # Example target position
ik_angles = niryo_one_chain.inverse_kinematics(target_position)

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

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# niryo_one_chain.plot(ik_angles, ax)    
# plt.show()

# loadlink from urdf file
urdf_file_path = "niryo_robot.urdf"
robot_chain_2 = Chain.from_urdf_file(urdf_file_path)

target_position_2 = [0.2, 0.2, 0.1]  # Example target position

# breakpoint()

angles_second = robot_chain_2.inverse_kinematics(target_position_2)

print("chain names:", robot_chain_2.name)
print("Inverse Kinematics Joint Angles 2:", angles_second)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
robot_chain_2.plot(angles_second, ax)
plt.show()

position_testify = robot_chain_2.forward_kinematics(angles_second)
print("Position for the angles: ", position_testify)


urdf_file_path = "niryo_arm.xml"