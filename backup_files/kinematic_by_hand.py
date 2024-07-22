from math import cos, sin, pi, atan2, sqrt
import numpy as np

class NiryoOneKinematics:
    def __init__(self, joint_ranges):
        # Assuming the distances and angles from the joint data in your XML
        self.d1 = 0.203   # Distance from base to shoulder
        self.a1 = 0       # No offset in the x direction from joint 1 to 2
        self.alpha1 = -pi/2  # Rotation around x from joint 1 to 2
        
        self.d2 = 0.08    # Distance from shoulder to elbow along z-axis
        self.a2 = 0.21    # Arm link length
        self.alpha2 = 0   # No rotation around x from joint 2 to 3

        self.d3 = 0       # Elbow operates at the intersection
        self.a3 = 0.0415 + 0.03  # Forearm link length
        self.alpha3 = pi/2  # Rotation around x from joint 3 to 4

        self.d4 = 0.19    # Distance along z-axis from forearm to wrist
        self.a4 = 0       # No offset in the x direction from joint 4 to 5
        self.alpha4 = -pi/2  # Rotation around x from joint 4 to 5

        self.d5 = 0       # Wrist operates at the intersection
        self.a5 = 0.0164  # Hand link length
        self.alpha5 = pi/2  # Rotation around x from joint 5 to 6

        self.d6 = 0.0203 + 0.0453  # Length to the gripper center
        self.a6 = 0       # No offset in the x direction from joint 6 to gripper
        self.alpha6 = 0   # No rotation around x from joint 6 to gripper center
        
        # joint ranges for each joint
        self.joint_ranges = joint_ranges

    def dh_transform(self, theta, d, a, alpha):
        return np.array([
            [cos(theta), -sin(theta) * cos(alpha),  sin(theta) * sin(alpha), a * cos(theta)],
            [sin(theta),  cos(theta) * cos(alpha), -cos(theta) * sin(alpha), a * sin(theta)],
            [0,            sin(alpha),              cos(alpha),              d],
            [0,            0,                      0,                       1]
        ])

    def forward_kinematics(self, joint_angles):
        T = np.eye(4)
        params = [
            (joint_angles[0], self.d1, self.a1, self.alpha1),
            (joint_angles[1], self.d2, self.a2, self.alpha2),
            (joint_angles[2], self.d3, self.a3, self.alpha3),
            (joint_angles[3], self.d4, self.a4, self.alpha4),
            (joint_angles[4], self.d5, self.a5, self.alpha5),
            (joint_angles[5], self.d6, self.a6, self.alpha6)
        ]
        for theta, d, a, alpha in params:
            T = np.dot(T, self.dh_transform(theta, d, a, alpha))
        return T
    
robot = NiryoOneKinematics(joint_ranges=np.array([
    [-2.949, 2.949],
    [-1.134, 1.483],
    [-2.443, 2.443],
    [-1.151, 1.151],
    [-2.443, 2.443],
    [-1.151, 1.151]
]))

target_angles = [0.00369, -0.0135, -0.511, -1.48, -0.000, 0.901]
forward_kinematics = robot.forward_kinematics(target_angles)
print("End-Effector Position:", forward_kinematics[:3, 3])