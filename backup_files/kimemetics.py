import numpy as np
from scipy.optimize import minimize

class NiryoOneKinematics:
    def __init__(self):
        self.joint_limits = [
            (-3.05433, 3.05433),  # radians
            (-1.5708, 0.640187),
            (-1.39749, 1.5708),
            (-3.05433, 3.05433),
            (-1.74533, 1.91986),
            (-2.57436, 2.57436)
        ]
        self.link_lengths = {
            'shoulder_to_elbow': 0.21,
            'elbow_to_wrist': 0.18,
            'wrist_to_hand': 0.05
        }

    def rotation_matrix_z(self, theta):
        return np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1]
        ])

    def transformation_matrix(self, joint_angle, d, a):
        """ Creates a transformation matrix for rotation around Z and translation """
        Rz = self.rotation_matrix_z(joint_angle)
        T = np.eye(4)
        T[:3, :3] = Rz
        T[2, 3] = d
        T[0, 3] = a
        return T

    def forward_kinematics(self, joint_angles):
        T = np.eye(4)
        chain_lengths = [0, self.link_lengths['shoulder_to_elbow'], 0.08, 
                         self.link_lengths['elbow_to_wrist'], 0, self.link_lengths['wrist_to_hand'], 0]
        for i, angle in enumerate(joint_angles):
            T = T @ self.transformation_matrix(angle, chain_lengths[i], 0)
        end_effector_pos = T[:3, 3]
        return end_effector_pos

    def target_joint_position(self, joint_angles, joint_index):
        """ Calculate the position of any specific joint """
        T = np.eye(4)
        chain_lengths = [0, self.link_lengths['shoulder_to_elbow'], 0.08, 
                         self.link_lengths['elbow_to_wrist'], 0, self.link_lengths['wrist_to_hand'], 0]
        for i, angle in enumerate(joint_angles):
            T = T @ self.transformation_matrix(angle, chain_lengths[i], 0)
            if i == joint_index:
                break
        joint_position = T[:3, 3]
        return joint_position

    def inverse_kinematics(self, target_position, joint_index, initial_guess):
        """ Use numerical optimization to find the joint angles that achieve the target position for joint 5 """
        def objective(joint_angles):
            pos = self.target_joint_position(joint_angles, joint_index)
            return np.linalg.norm(pos - target_position)

        bounds = self.joint_limits
        result = minimize(objective, initial_guess, bounds=bounds)
        if result.success:
            return result.x
        else:
            raise ValueError("Optimization failed: " + result.message)

# Example usage:
kinematics = NiryoOneKinematics()
target_position = np.array([1, 1, 1])  # Target position for joint 5
initial_guess = [0, 0, 0, 0, 0, 0]  # Initial guess of joint angles
joint_index = 5

joint_angles = kinematics.inverse_kinematics(target_position, joint_index, initial_guess)
print("Calculated Joint Angles:", joint_angles)
