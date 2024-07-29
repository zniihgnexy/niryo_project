import math
import time
import mujoco
import numpy as np
from GradientDescentIK import GradientDescentIK
from LevenbergMarquardtIK import LevenbergMarquardtIK

def get_target_angles(model, data, step_size, tol, alpha, jacp, jacr, movable_joints_indices, target_position, initialize_angles):
    body_id = model.body('g1_mainSupport_link').id
    ik = GradientDescentIK(model, data, step_size, tol, alpha, jacp, jacr, movable_joints_indices)
    ik_lm = LevenbergMarquardtIK(model, data, step_size, tol, alpha, 0.1, movable_joints_indices)
    target_angles = ik_lm.calculate(target_position, initialize_angles[:8], body_id)
    return target_angles

def pre_calculation(target_position, task_name, robot_controller, initialize_angles):
    print(f"Calculating target angles for task: {task_name}")
    target_angles = get_target_angles(robot_controller.model, robot_controller.data, robot_controller.step_size, robot_controller.tol, robot_controller.alpha, robot_controller.jacp, robot_controller.jacr, robot_controller.movable_joints_indices, target_position, initialize_angles)
    print(f"Target angles for {task_name}: {target_angles}")
    return target_angles

def step_move_to_position(robot_controller, target_position, initialize_angles, viewer):
    target_angles = robot_controller.get_target_angles(target_position, initialize_angles)
    robot_controller.control_arm_to_position(target_angles)
    robot_controller.sync_viewer(viewer)
    time.sleep(0.02)

def step_close_gripper(robot_controller, viewer, data, target_position, FLAG):
    robot_controller.control_arm_to_position_nogripper(robot_controller.joint_indices)
    end_effector_pos = data.body(robot_controller.model.body('g1_mainSupport_link').id).xpos 
    error_tolerance = math.sqrt((end_effector_pos[0] - target_position[0])**2 + (end_effector_pos[1] - target_position[1])**2 + (end_effector_pos[2] - target_position[2])**2)
    if FLAG == 0 and error_tolerance < 0.01:
        FLAG = True
    mujoco.mj_step(robot_controller.model, robot_controller.data)
    robot_controller.sync_viewer(viewer)
    time.sleep(0.02)

def step_lift_up(robot_controller, target_position, initialize_angles, steps, viewer):
    target_angles = robot_controller.get_target_angles(target_position, initialize_angles)
    for _ in range(steps):
        robot_controller.control_arm_to_position(target_angles)
        robot_controller.sync_viewer(viewer)
        time.sleep(0.02)

def step_release_gripper(robot_controller, steps, viewer):
    for _ in range(steps):
        robot_controller.control_arm_to_position(robot_controller.joint_indices)
        robot_controller.sync_viewer(viewer)
        time.sleep(0.02)

def set_mocap_position(model, data, body_name, position):
    """Set the position of a mocap body."""
    body_idx = model.body(name=body_name).id
    mocap_idx = model.body_mocapid[body_idx]
    if mocap_idx == -1:
        raise ValueError("Body is not a mocap body or does not exist.")
    
    data.mocap_pos[mocap_idx][:] = np.array(position)

def has_contact(data, sensor_name, model):
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    return data.sensor(sensor_id).data[0] > 0 

def lift_to_new_position(robot_controller, initial_position, lift_height, initialize_angles):
    target_position = initial_position + np.array([0, 0, lift_height])
    target_angles = robot_controller.get_target_angles(target_position, initialize_angles)
    return target_angles

def run_simulation(robot, viewer):
    initial_ball_position = robot.get_ball_position()
    target_position = initial_ball_position + np.array([0, 0, 0.1])  # Move up 10cm
    target_angles = robot.get_target_angles(target_position)
    robot.control_arm_to_position(target_angles)
    robot.sync_viewer(viewer)
