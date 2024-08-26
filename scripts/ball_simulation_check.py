import mujoco
import numpy as np
from robot_controller import RobotController

def updates_gripper_angles(niryo, target_angles, FLAG, joint_indices):
    if FLAG == 1:
        target_angles = RobotController.control_close_gripper(niryo, target_angles, joint_indices)
    else:
        target_angles = RobotController.control_open_gripper(niryo, target_angles, joint_indices)
    return target_angles

def check_before_close_gripper(data, body_id, target_position, tolerance):
    end_effector_pos = data.body(body_id).xpos
    error_tolerance = np.linalg.norm(end_effector_pos - target_position)
    return error_tolerance < tolerance

def ball_pos_update(data, site_id):
    return data.site(site_id).xpos

def get_gripper_position(data, body_id):
    return data.body(body_id).xpos