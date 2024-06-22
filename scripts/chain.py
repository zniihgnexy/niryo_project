from ikpy.chain import Chain

# Load the URDF directly into an IKpy chain
robot_chain = Chain.from_urdf_file("./niryo_two.urdf")

# Example target position for the end effector
target_position = [0.3, 0.3, 0.3]  # Modify this as needed

# Calculate inverse kinematics to get the joint angles for reaching the target position
ik_angles = robot_chain.inverse_kinematics(target_position)

print("Inverse Kinematics Result:", ik_angles)
# breakpoint()
# get teh angle value that i should send to the robot
# ik_angles = [ik_angles[joint] for joint in robot_chain.joint_names]
# print("Inverse Kinematics Result:", ik_angles)

# Calculate forward kinematics using the obtained joint angles to verify the position
joint_position = robot_chain.forward_kinematics(ik_angles)

# Extract the position of the end effector from the transformation matrix
end_effector_position = joint_position[:3, 3]

print("Forward Kinematics Result:", end_effector_position)



import matplotlib.pyplot as plt

# ax = niryo_one_chain.plot(ik_angles)
plt.figure()
ax = plt.axes(projection='3d')
ax.set_box_aspect([1,1,1])  # Aspect ratio is 1:1:1
robot_chain.plot(ik_angles, ax)

# Plot the target position
ax.scatter(target_position[0], target_position[1], target_position[2], c='r', marker='x', label='Target Position')

# Plot the end effector position
ax.scatter(end_effector_position[0], end_effector_position[1], end_effector_position[2], c='g', marker='o', label='End Effector Position')

plt.legend()
plt.title('Niryo One Robot Arm')
plt.show()
