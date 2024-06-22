from ikpy.chain import Chain

class Kinematic:
    def __init__(self, urdf_path):
        self.robot_chain = Chain.from_urdf_file(urdf_path)

    def inverse_kinematics(self, target_position):
        return self.robot_chain.inverse_kinematics(target_position)

    def forward_kinematics(self, ik_angles):
        return self.robot_chain.forward_kinematics(ik_angles)

    def plot(self, ik_angles, target_position):
        import matplotlib.pyplot as plt

        plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_box_aspect([1,1,1])  # Aspect ratio is 1:1:1
        self.robot_chain.plot(ik_angles, ax)

        # Plot the target position
        ax.scatter(target_position[0], target_position[1], target_position[2], c='r', marker='x', label='Target Position')
        joint_position = self.robot_chain.forward_kinematics(ik_angles)

        # Extract the position of the end effector from the transformation matrix
        end_effector_position = joint_position[:3, 3]
        ax.scatter(end_effector_position[0], end_effector_position[1], end_effector_position[2], c='g', marker='o', label='End Effector Position')

        plt.legend()
        plt.title('Niryo One Robot Arm')
        plt.show()

# import matplotlib.pyplot as plt

# # Usage
# target_position = [0.1, 0.1, 0.1]  # Modify this as needed
# kinematic = Kinematic("./niryo_two.urdf")
# ik_angles = kinematic.inverse_kinematics(target_position)
# print("Inverse Kinematics Result:", ik_angles)

# kinematic.plot(ik_angles, target_position)