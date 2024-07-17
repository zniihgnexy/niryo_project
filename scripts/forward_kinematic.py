import numpy as np

def dh_transform(theta, d, a, alpha):
    """
    Create the transformation matrix using Denavit-Hartenberg parameters.
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0, sa, ca, d],
        [0, 0, 0, 1]
    ])

def jacobian(dh_params, joint_angles):
    """
    Compute the Jacobian matrix for the given joint angles.
    """
    num_joints = len(joint_angles)
    J = np.zeros((6, num_joints))  # Adjusted to include angular components
    T = np.eye(4)
    T_i = [T]  # Store transformations for each joint

    for i, params in enumerate(dh_params):
        T = T @ dh_transform(joint_angles[i], **params)
        T_i.append(T)

    T_end = T_i[-1]  # Final transformation matrix
    O_n = T_end[:3, 3]  # Position of the end-effector

    for i in range(num_joints):
        T = T_i[i]
        z = T[:3, 2]  # Rotation axis for the joint
        p = O_n - T[:3, 3]
        J[:3, i] = np.cross(z, p)  # Linear velocity component
        J[3:, i] = z  # Angular velocity component

    return J

class RobotArm:
    def __init__(self, joint_ranges, gripper_length=0):
        self.joint_ranges = joint_ranges
        self.gripper_length = gripper_length
        self.dh_params = [
            {'d': 0.203, 'a': 0, 'alpha': 0},  # Base to Shoulder: translation along z of 0.103m, no rotation about common normal (x-axis)
            {'d': 0.08, 'a': 0, 'alpha': np.pi/2},  # Shoulder to Arm: translation along x of 0.08m, rotation of +pi/2 about x-axis
            {'d': 0, 'a': 0.21, 'alpha': 0},  # Arm to Elbow: translation along x of 0.21m, no twist
            {'d': 0, 'a': 0.0415, 'alpha': -np.pi/2},  # Elbow to Forearm: translation along x of 0.0415m, rotation of -pi/2 about x-axis
            {'d': 0.19, 'a': 0, 'alpha': np.pi/2},  # Forearm to Wrist: translation along z of 0.19m, rotation of +pi/2 about x-axis
            {'d': 0, 'a': 0.0164, 'alpha': 0},  # Wrist to Hand: translation along x of 0.0164m, no twist
            {'d': 0.0, 'a': 0, 'alpha': -np.pi}  # Additional fixed joint: translation along z of 0.0203m, rotation of -pi about x-axis (effectively flipping direction)
        ]


    def forward_kinematics(self, joint_angles):
        T = np.eye(4)
        positions = [T[:3, 3]]  # Starting position at the base
        for i, (params, angle) in enumerate(zip(self.dh_params, joint_angles)):
            T_next = dh_transform(angle, **params)
            T = T @ T_next
            positions.append(T[:3, 3])  # Store the position of the end of each joint
        return T, np.array(positions)

    def inverse_kinematics(self, target_position, initial_angles=None, tolerance=1e-5, max_iterations=500):
        joint_angles = np.array(initial_angles if initial_angles is not None else [0] * len(self.dh_params), dtype=np.float64)

        for _ in range(max_iterations):
            T, joint_position = self.forward_kinematics(joint_angles)
            position_error = target_position - T[:3, 3]
            if np.linalg.norm(position_error) < tolerance:
                return joint_angles

            J = jacobian(self.dh_params, joint_angles)
            delta_theta = np.linalg.pinv(J[:3, :]) @ position_error
            joint_angles += delta_theta
            joint_angles = np.clip(joint_angles, self.joint_ranges[:, 0], self.joint_ranges[:, 1])

        raise ValueError("Inverse kinematics did not converge")
    
    def rotation_matrix_to_euler_angles(self, R):
        """
        Convert a rotation matrix to Euler angles (roll, pitch, yaw)
        """
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0

        return np.array([x, y, z])

    def get_end_effector_orientation(self, joint_angles):
        """
        Get the end effector's orientation in terms of Euler angles (roll, pitch, yaw)
        """
        T = self.forward_kinematics(joint_angles)
        R = T[:3, :3]
        return self.rotation_matrix_to_euler_angles(R)

# Setup example
joint_ranges = np.array([
    [-2.949, 2.949],
    [-1.83, 0.61],
    [-1.397485, 1.57079632679],
    [-2.089, 2.089],
    [-1.04533, 0.2397],
    [-1.57436, 1.57436]
])

robot = RobotArm(joint_ranges, gripper_length=0)
initial_angles = [1, 0, 0, 0, 0, 0, 0]

# joint 6 joint position, correct version test
target_position = np.array([0.15828916, 0.00081683, 0.19676705])
try:
    calculated_angles = robot.inverse_kinematics(target_position, initial_angles)
    print("Calculated Joint Angles:", calculated_angles)
except ValueError as e:
    print(e)

target_angles = [0.00369, -0.0135, -0.511, -1.48, -0.000, 0.901, 0]
T, joint_positions  = robot.forward_kinematics(initial_angles)
print("End-Effector Position:", joint_positions)

# euler_angles = robot.get_end_effector_orientation(target_angles)
# print("End-Effector Orientation (Euler angles):", euler_angles)

# gripper_length = 0.0656
# gripper_shift = np.array([gripper_length*np.cos(euler_angles[2]), gripper_length*np.sin(euler_angles[2]), 0])
# print("Gripper Position:", forward_kinematics[:3, 3] + gripper_shift)