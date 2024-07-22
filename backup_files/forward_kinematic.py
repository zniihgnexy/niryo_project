import numpy as np
import matplotlib.pyplot as plt

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

    for i, (params, angle) in enumerate(zip(dh_params, joint_angles)):
        T = T @ dh_transform(angle + params['theta_adjustment'], params['d'], params['a'], params['alpha'])
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
            {'theta_adjustment': 0, 'd': 0.08, 'a': 0, 'alpha': -np.pi/2},  # Base to Shoulder
            {'theta_adjustment': -np.pi/2, 'd': 0, 'a': 0.21, 'alpha': np.pi},  # Shoulder to Arm
            {'theta_adjustment': np.pi, 'd': 0, 'a': 0.0415, 'alpha': np.pi/2},  # Arm to Elbow
            {'theta_adjustment': 0, 'd': 0.19, 'a': 0, 'alpha': -np.pi/2},  # Elbow to Forearm
            {'theta_adjustment': 0, 'd': 0, 'a': 0.0164, 'alpha': np.pi/2},  # Forearm to Wrist
            {'theta_adjustment': np.pi, 'd': 0.0203, 'a': 0, 'alpha': 0}  # Wrist to Hand
        ]

    def forward_kinematics(self, joint_angles):
        T = np.eye(4)
        positions = [T[:3, 3]]  # Starting position at the base
        for i, (params, angle) in enumerate(zip(self.dh_params, joint_angles)):
            T_next = dh_transform(angle + params['theta_adjustment'], params['d'], params['a'], params['alpha'])
            T = T @ T_next
            positions.append(T[:3, 3].copy())  # Store the position of the end of each joint
        return T, positions

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
        T, _ = self.forward_kinematics(joint_angles)
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
initial_angles = [0, 0, 0, 0, 0, 0]  # All angles are set to zero

# Joint 6 joint position, correct version test
target_position = np.array([0.15828916, 0.00081683, 0.19676705])
try:
    calculated_angles = robot.inverse_kinematics(target_position, initial_angles)
    print("Calculated Joint Angles:", calculated_angles)
except ValueError as e:
    print(e)

target_angles = [0.00369, -0.0135, -0.511, -1.48, -0.000, 0.901]
T, joint_positions  = robot.forward_kinematics(target_angles)
print("End-Effector Position:", joint_positions)

# Euler angles for the end effector orientation
euler_angles = robot.get_end_effector_orientation(target_angles)
print("End-Effector Orientation (Euler angles):", euler_angles)

# Plotting the robot structure
def plot_robot_structure(joint_positions):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    joint_positions = np.array(joint_positions)
    ax.plot(joint_positions[:, 0], joint_positions[:, 1], joint_positions[:, 2], '-o', color='blue')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()

plot_robot_structure(joint_positions)
