import mujoco
import numpy as np
import time
from mujoco import viewer
import matplotlib.pyplot as plt
from pid_controller import PIDController

# Load the model
model_path = '/home/xz2723/niryo_project/meshes/mjmodel.xml'
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Joint names and initialization
joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'left_clamp_joint', 'right_clamp_joint']
joint_indices = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names}

fixed_positions = {
    'joint_1': 0.00369,
    'joint_2': -0.0135,
    'joint_3': -0.511,
    'joint_4': -1.48,
    'joint_5': -0.000,
    'joint_6': 0.901,
    'left_clamp_joint': -0.00000,
    'right_clamp_joint': 0.00036
}

initialize_angles = np.array([fixed_positions[name] for name in joint_names])

target_position = [0.00000, 0.25000, 0.2000]

# Define the inverse kinematics class using gradient descent
class GradientDescentIK:
    def __init__(self, model, data, step_size, tol, alpha, jacp, jacr, movable_joints_indices):
        self.model = model
        self.data = data
        self.step_size = step_size
        self.tol = tol
        self.alpha = alpha
        self.jacp = jacp
        self.jacr = jacr
        self.movable_joints_indices = movable_joints_indices
    
    def check_joint_limits(self, q):
        """Check if the joints is under or above its limits"""
        for i in range(len(q)):
            q[i] = max(self.model.jnt_range[i][0], min(q[i], self.model.jnt_range[i][1]))

    def calculate(self, goal, init_q, body_id):
        """Calculate the desired joints angles for the goal"""
        self.data.qpos[self.movable_joints_indices] = init_q
        mujoco.mj_forward(self.model, self.data)
        current_pose = self.data.body(body_id).xpos
        error = np.subtract(goal, current_pose)

        while (np.linalg.norm(error) >= self.tol):
            # Calculate Jacobian
            mujoco.mj_jac(self.model, self.data, self.jacp, self.jacr, current_pose, body_id)
            # Calculate gradient
            grad = self.alpha * self.jacp[:, self.movable_joints_indices].T @ error
            # Compute next step
            self.data.qpos[self.movable_joints_indices] += self.step_size * grad
            # Check joint limits
            self.check_joint_limits(self.data.qpos[self.movable_joints_indices])
            # Compute forward kinematics
            mujoco.mj_forward(self.model, self.data) 
            # Calculate new error
            current_pose = self.data.body(body_id).xpos
            error = np.subtract(goal, current_pose)
        return self.data.qpos[self.movable_joints_indices].copy()

# Initialize variables for inverse kinematics
body_id = model.body('hand_link').id  # end-effector ID
jacp = np.zeros((3, model.nv))  # Translational Jacobian
jacr = np.zeros((3, model.nv))  # Rotational Jacobian
goal = target_position  # Desired position
step_size = 0.01
tol = 0.001
alpha = 0.5
init_q = initialize_angles

movable_joints_indices = [joint_indices[joint_name] for joint_name in joint_names[:]]  # Exclude clamp joints

ik = GradientDescentIK(model, data, step_size, tol, alpha, jacp, jacr, movable_joints_indices)

# Get desired joint angles using inverse kinematics
target_angles = ik.calculate(goal, init_q, body_id)
print("Target angles:", target_angles)

# breakpoint()

# Initialize PID controllers
pids = {
    'joint_1': PIDController(100, 0.5, 100),
    'joint_2': PIDController(100, 0.8, 100),
    'joint_3': PIDController(100, 0.06, 100),
    'joint_4': PIDController(180, 0.06, 100),
    'joint_5': PIDController(100, 0.0001, 150),
    'joint_6': PIDController(162.5, 0.06, 100),
    'left_clamp_joint': PIDController(10, 0.0001, 5),
    'right_clamp_joint': PIDController(10, 0.0001, 5)
}

joint_angle_history = {name: [] for name in joint_names}
position_updates = []

# Function to control arm position
def control_arm_to_position(model, data, joint_names, joint_indices, pids, target_angles):
    for name in joint_names:
        joint_index = joint_indices[name]
        current_position = data.qpos[joint_index]
        target_angle = target_angles[joint_indices[name]]
        control_signal = pids[name].calculate(target_angle, current_position)
        data.ctrl[joint_index] = control_signal
        joint_angle_history[name].append(current_position)

# Simulation loop for synchronized movement
with viewer.launch_passive(model, data) as Viewer:
    Viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 0
    Viewer.sync()

    # Simulation parameters
    duration = 10  # seconds
    steps = int(duration * 50)  # Assuming 50 Hz simulation frequency

    # Set initial joint positions
    for name, angle in fixed_positions.items():
        joint_index = joint_indices[name]
        data.qpos[joint_index] = angle

    Viewer.sync()
    time.sleep(0.02)

    print("Starting simulation...")
    for step in range(steps):
        # Update control signals for all joints at each step
        control_arm_to_position(model, data, joint_names, joint_indices, pids, target_angles)
        
        end_effector = data.body(body_id).xpos
        position_updates.append(end_effector)
        
        mujoco.mj_step(model, data)
        Viewer.sync()
        time.sleep(0.02)  # Sleep to match the assumed simulation frequency

    print("All joints have moved towards their target positions.")
    print("End-effector final position:", end_effector)

# Plot joint angles over time
fig, axs = plt.subplots(4, 2, figsize=(15, 10))
for idx, name in enumerate(joint_names):
    ax = axs[idx // 2, idx % 2]
    ax.plot(np.linspace(0, duration, len(joint_angle_history[name])), joint_angle_history[name])
    ax.set_title(f'Trajectory of {name}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (rad)')

plt.tight_layout()
plt.show()

# Print final joint angles
end_angles = [data.qpos[joint_indices[name]] for name in joint_names]
print("Final joint angles:", end_angles)
