import numpy as np

def dh_transform(theta, d, a, alpha):
    """
    Create the transformation matrix using Denavit-Hartenberg parameters.

    :param theta: Joint angle (rotation around z-axis)
    :param d: Offset along z-axis
    :param a: Offset along x-axis (link length)
    :param alpha: Link twist (rotation around x-axis)
    :return: Transformation matrix as a NumPy array
    """
    T = np.array([
        [np.cos(theta), -np.sin(theta) * np.cos(alpha),  np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
        [np.sin(theta),  np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
        [0,              np.sin(alpha),                  np.cos(alpha),                d],
        [0,              0,                              0,                            1]
    ])
    return T

def gripper_center_transform(length):
    """
    Compute the transformation matrix to the gripper center.

    :param length: Length from the last joint to the center of the gripper along z-axis
    :return: 4x4 transformation matrix as a NumPy array
    """
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, length / 2],  # Assuming the gripper extends equally from the joint
        [0, 0, 0, 1]
    ])

class RobotArm:
    def __init__(self, gripper_length=0.1):  # default gripper length
        self.gripper_length = gripper_length
        self.dh_params = [
            {'theta': 0, 'd': 0.03, 'a': 0, 'alpha': np.pi/2},
            {'theta': -np.pi/2, 'd': 0, 'a': 0.08, 'alpha': 0},
            {'theta': 0, 'd': 0, 'a': 0.21, 'alpha': np.pi/2},
            {'theta': np.pi/2, 'd': 0.0415, 'a': 0.03, 'alpha': -np.pi/2},
            {'theta': -np.pi/2, 'd': 0, 'a': 0.19, 'alpha': np.pi/2},
            {'theta': np.pi/2, 'd': 0.0164, 'a': 0, 'alpha': 0},
        ]

    def forward_kinematics(self, joint_angles):
        T = np.eye(4)  # Start with the identity matrix
        for idx, angle in enumerate(joint_angles):
            params = self.dh_params[idx]
            params['theta'] = angle
            T = np.dot(T, dh_transform(**params))
        T = np.dot(T, gripper_center_transform(self.gripper_length))
        return T


# Example usage:
robot = RobotArm(gripper_length=0.0475)  # Adjust this length based on your model
# target_angles = [np.pi/6, -np.pi/4, np.pi/3, -np.pi/6, np.pi/2, -np.pi/3]  # Sample target joint angles
target_angles = [0.0000, -0.0135, -0.511, -1.48, -0.000, 0.901]  # Sample target joint angles
end_effector_transform = robot.forward_kinematics(target_angles)
print("Transformation Matrix of End-Effector at Gripper Center:")
print(end_effector_transform)

print("end effector position:", end_effector_transform[:3, 3])