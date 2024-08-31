import mujoco
"""
This module contains the `RobotController` class which is responsible for controlling the robot's movements and actions.
Attributes:
    model (mujoco.Model): The Mujoco model of the robot.
    data (mujoco.Data): The Mujoco data associated with the model.
    joint_names (list): A list of the names of the robot's joints.
    pid_params (dict): A dictionary containing the PID parameters for each joint.
    joint_indices (dict): A dictionary mapping joint names to their corresponding indices.
    pids (dict): A dictionary containing PIDController objects for each joint.
    position_updates (list): A list of position updates of the robot's end effector.
    joint_angle_history (dict): A dictionary mapping joint names to lists of their angle history.
    movable_joints_indices (list): A list of indices of the movable joints.
    jacp (numpy.ndarray): A 3xnv array representing the Jacobian matrix for position control.
    jacr (numpy.ndarray): A 3xnv array representing the Jacobian matrix for rotation control.
    alpha (float): The learning rate for the IK algorithms.
    tol (float): The tolerance for convergence in the IK algorithms.
    step_size (float): The step size for the IK algorithms.
    site_id (int): The ID of the gripper center site.
    sensor_id (int): The ID of the touch sensor.
    end_effector_pos (numpy.ndarray): The position of the robot's end effector.
Methods:
    get_target_angles(target_position, initialize_angles):
        Calculates the target joint angles for a given target position.
    control_arm_to_position(target_angles):
        Controls the robot's arm to reach the target joint angles.
    control_close_gripper(target_angles, joint_indices):
        Controls the robot's gripper to close.
    control_open_gripper(target_angles, joint_indices):
        Controls the robot's gripper to open.
    sync_viewer(viewer):
        Synchronizes the viewer with the robot's model and data.
    set_initial_positions(fixed_positions, joint_indices):
        Sets the initial positions of the robot's joints.
    quaternion_to_euler(w, x, y, z):
        Converts a quaternion to Euler angles.
    set_mocap_position(body_name, position):
        Sets the position of a mocap body.
"""
import numpy as np
from GradientDescentIK import GradientDescentIK
from LevenbergMarquardtIK import LevenbergMarquardtIK
from pid_controller import PIDController, PIDControllerWithDerivativeFilter



class RobotController:
    def __init__(self, model, data, joint_names, pid_params, joint_indices):
        self.model = model
        self.data = data
        self.joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'left_clamp_joint', 'right_clamp_joint']
        self.joint_indices = joint_indices
        self.pids = {
            'joint_1': PIDController(100, 0, 100),
            'joint_2': PIDController(100, 0, 100),
            'joint_3': PIDController(100, 0, 100),
            'joint_4': PIDController(180, 0, 100),
            'joint_5': PIDController(100, 0, 100),
            'joint_6': PIDController(162.5, 0, 100),
            'left_clamp_joint': PIDControllerWithDerivativeFilter(20, 0, 100),
            'right_clamp_joint': PIDControllerWithDerivativeFilter(20, 0, 100)
        }
        self.position_updates = []
        self.joint_angle_history = {name: [] for name in joint_names}
        self.movable_joints_indices = [self.joint_indices[joint_name] for joint_name in joint_names[:]]
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))
        self.alpha = 0.5
        self.tol = 0.001
        self.step_size = 0.01
        self.site_id = model.site('gripper_center').id
        self.sensor_id = model.sensor('touch').id
        self.end_effector_pos = data.site(self.site_id).xpos

    def get_target_angles(self, target_position, initialize_angles):
        body_id = self.model.body('g1_mainSupport_link').id
        ik = GradientDescentIK(self.model, self.data, self.step_size, self.tol, self.alpha, self.jacp, self.jacr, self.movable_joints_indices)
        ik_lm = LevenbergMarquardtIK(self.model, self.data, self.step_size, self.tol, self.alpha, 0.1, self.movable_joints_indices)
        target_angles = ik_lm.calculate(target_position, initialize_angles[:8], body_id)
        return target_angles

    def control_arm_to_position(self, target_angles):
        for name in self.joint_names:
            joint_indices = self.joint_indices
            joint_index = joint_indices[name]
            current_position = self.data.qpos[joint_index]
            target_angle = target_angles[joint_index]
            control_signal = self.pids[name].calculate(target_angle, current_position)
            self.data.ctrl[joint_index] = control_signal
            self.joint_angle_history[name].append(current_position)
        self.position_updates.append(self.data.site(self.site_id).xpos)

    def control_close_gripper(self, target_angles, joint_indices):
        for name in ['left_clamp_joint', 'right_clamp_joint']:
            joint_index = joint_indices[name]
            if name == 'left_clamp_joint':
                target_angles[joint_index] = -0.00505
            else:
                target_angles[joint_index] = 0.00505
        return target_angles
    
    def control_open_gripper(self, target_angles, joint_indices):
        for name in ['left_clamp_joint', 'right_clamp_joint']:
            joint_index = joint_indices[name]
            if name == 'left_clamp_joint':
                target_angles[joint_index] = 0.00505
            else:
                target_angles[joint_index] = -0.00505
        return target_angles

    def sync_viewer(self, viewer):
        mujoco.mj_step(self.model, self.data)
        viewer.sync()

    def set_initial_positions(self, fixed_positions, joint_indices):
        for name, angle in fixed_positions.items():
            joint_index = joint_indices[name]
            self.data.qpos[joint_index] = angle

    def quaternion_to_euler(self, w, x, y, z):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch = np.arcsin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)

        return roll, pitch, yaw

    def set_mocap_position(self, body_name, position):
        # print("set mocap position")
        body_idx = self.model.body(name=body_name).id
        mocap_idx = self.model.body_mocapid[body_idx]
        if mocap_idx == -1:
            raise ValueError("Body is not a mocap body or does not exist.")
        self.data.mocap_pos[mocap_idx][:] = np.array(position)
