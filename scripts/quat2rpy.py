import numpy as np

def quaternion_to_rpy(quat):
    """
    Convert a quaternion into roll, pitch, yaw angles.
    quat: [x, y, z, w]
    """
    x, y, z, w = quat
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = np.arcsin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return roll, pitch, yaw


quat_arm = [0.5, 0.5, -0.5, 0.5]
quat_elbow = [0.707107, 0, 0, -0.707107]
quat_forearm = [0.707107, 0, 0.707107, 0]
quat_wrist = [0.707107, 0, -0.707107, 0]
quat_hand = [0.707107, 0, 0.707107, 0]
quat_gripper_left = [-0.499999, 0.499999, 0.500001, -0.500001]
quat_gripper_right = [-0.499999, 0.499999, 0.500001, -0.500001]

print("Roll, pitch, yaw for arm:", quaternion_to_rpy(quat_arm))
print("Roll, pitch, yaw for elbow:", quaternion_to_rpy(quat_elbow))
print("Roll, pitch, yaw for forearm:", quaternion_to_rpy(quat_forearm))
print("Roll, pitch, yaw for wrist:", quaternion_to_rpy(quat_wrist))
print("Roll, pitch, yaw for hand:", quaternion_to_rpy(quat_hand))
print("Roll, pitch, yaw for gripper left:", quaternion_to_rpy(quat_gripper_left))
print("Roll, pitch, yaw for gripper right:", quaternion_to_rpy(quat_gripper_right))

