import mujoco
import numpy as np
import time
from mujoco import viewer

# Assume these imports are available for KDL usage
from PyKDL import Chain, ChainFkSolverPos_recursive, ChainIkSolverPos_LMA, Frame, Vector, Rotation

# PID Controller Class
class PIDController:
    def __init__(self, kp, ki, kd, setpoint=0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.previous_error = 0
        self.integral = 0
        self.dt = 0.02  # Assuming control at 50 Hz

    def update(self, current_value):
        error = self.setpoint - current_value
        self.integral += error * self.dt
        derivative = (error - self.previous_error) / self.dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error
        return output

# Initialize KDL Solver
def initialize_kdl_chain():
    chain = Chain()

    return chain

def compute_kdl_ik(chain, target_pos):
    # Create solver
    fk_solver = ChainFkSolverPos_recursive(chain)
    ik_solver = ChainIkSolverPos_LMA(chain)

    # Target Frame: modify according to actual use-case
    target_frame = Frame(Rotation.RPY(0, 0, 0), Vector(target_pos[0], target_pos[1], target_pos[2]))

    # Initial joint positions: np.zeros(chain.getNrOfJoints())
    q_out = JntArray(chain.getNrOfJoints())
    ik_solver.CartToJnt(initial_jnt_array, target_frame, q_out)

    return np.array([q_out[i] for i in range(q_out.rows())])

# Load the model
model_path = '/home/xz2723/niryo_project/meshes/mjmodel.xml'
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Initialize angles and controllers
initialize_angles = np.array([1, 0.5, 0.5, 0, 0, 0, 0, 0])
joint_controllers = {name: PIDController(1.0, 0.01, 0.5) for name in joint_names}

# Joint names
joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'left_clamp_joint', 'right_clamp_joint']
target_positions = [0.3, -0.3, 0.5]  # Desired XYZ position of the end-effector

# Map joint names to indices using mj_name2id
joint_indices = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names}

# Initialize joint positions according to target angles
for name in joint_names:
    joint_id = joint_indices[name]
    data.qpos[joint_id] = initialize_angles[joint_indices[name]]

# Initialize the passive viewer
with viewer.launch_passive(model, data) as Viewer:
    Viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 1
    Viewer.sync()

    # Hold initial position for 5 seconds
    start_time = time.time()
    while time.time() - start_time < 5:
        Viewer.sync()
        time.sleep(0.05)

    # KDL Setup
    chain = initialize_kdl_chain()
    target_joint_positions = compute_kdl_ik(chain, target_positions)

    # Simulation loop
    for step in range(int(5 * 50)):  # 5 seconds, 50 Hz
        for idx, name in enumerate(joint_names):
            joint_id = joint_indices[name]
            current_angle = data.qpos[joint_id]
            target_angle = target_joint_positions[idx]
            joint_controllers[name].setpoint = target_angle
            control_output = joint_controllers[name].update(current_angle)
            data.qpos[joint_id] += control_output

        mujoco.mj_step(model, data)
        Viewer.sync()
        time.sleep(0.02)  # 50 Hz

    print("Simulation ended.")
