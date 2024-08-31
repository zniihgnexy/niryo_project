import mujoco
"""
updates_gripper_angles(niryo, target_angles, FLAG, joint_indices)

Updates the gripper angles based on the given flag.

Parameters:
- niryo: The RobotController object representing the robot.
- target_angles: The current target angles of the gripper.
- FLAG: An integer flag indicating whether to close or open the gripper.
- joint_indices: A list of joint indices for the gripper.

Returns:
- The updated target angles of the gripper.

---

check_before_close_gripper(data, body_id, target_position, tolerance)

Checks if the end effector is close enough to the target position before closing the gripper.

Parameters:
- data: The mujoco data object.
- body_id: The ID of the body representing the end effector.
- target_position: The target position for the end effector.
- tolerance: The tolerance value for the distance between the end effector and the target position.

Returns:
- True if the distance between the end effector and the target position is less than the tolerance, False otherwise.

---

ball_pos_update(data, site_id)

Gets the position of a ball in the simulation.

Parameters:
- data: The mujoco data object.
- site_id: The ID of the site representing the ball.

Returns:
- The position of the ball.

---

get_gripper_position(data, body_id)

Gets the position of the gripper.

Parameters:
- data: The mujoco data object.
- body_id: The ID of the body representing the gripper.

Returns:
- The position of the gripper.
"""
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