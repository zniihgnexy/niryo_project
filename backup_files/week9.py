import mujoco
import numpy as np
import time
from mujoco import viewer
import matplotlib.pyplot as plt

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral_error = 0
        self.prev_error = 0
        self.dt = 0.02  # 控制周期，需要与模拟步长相匹配

    def calculate(self, target, current):
        error = target - current
        self.integral_error += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        output = (self.Kp * error) + (self.Ki * self.integral_error) + (self.Kd * derivative)
        self.prev_error = error
        return output


# Load the model
model_path = '/home/xz2723/niryo_project/meshes/mjmodel.xml'
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Initialize angles
initialize_angles = np.array([1.0000, 0.5000, 0.5000, 0.0000, 0.0000, 0.0000, 0.00000, 0.00000])

# Joint names and target angles
joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'left_clamp_joint', 'right_clamp_joint']
target_angles = {
    'joint_1': 1.0000, 
    'joint_2': 0.5000, 
    'joint_3': 1.0000, 
    'joint_4': 0.0000, 
    'joint_5': 0.0000, 
    'joint_6': 0.0000, 
    'left_clamp_joint': 0.00000, 
    'right_clamp_joint': 0.00000
}

# get the joint values from the target_angles dictionary
target_angles_values = np.array([target_angles[name] for name in joint_names])

# Map joint names to indices using mj_name2id
joint_indices = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names}

joint_angle_history = {name: [] for name in joint_names}

# Initialize joint positions according to target angles
for name in joint_names:
    joint_id = joint_indices[name]
    data.qpos[joint_id] = initialize_angles[joint_indices[name]]
    joint_angle_history[name].append(data.qpos[joint_id])

# breakpoint()

# Initialize the passive viewer
with viewer.launch_passive(model, data) as Viewer:
    # Set viewer settings, such as enabling wireframe mode
    Viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 1
    Viewer.sync()

    # Hold initial position
    initial_hold_time = 1  # seconds
    start_time = time.time()
    print("Holding initial position for 2 seconds...")
    while time.time() - start_time < initial_hold_time:
        Viewer.sync()
        time.sleep(0.05)
    print("Initial position hold complete.")

    duration = 2
    steps = int(duration * 50)  # Assuming 50 Hz simulation frequency
    time_step = 1 / 5

    # kp, kv, ki = 200, 20, 0.1
    # set the PID gains in three arrays for each joint
    # first tuning kp, range 0-200
    kp = np.array([179, 180, 180, 180, 173, 173, 0.25, 0.25])
    kv = np.array([33, 33, 33, 33, 20, 20, 1, 1])
    ki = np.array([0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.001, 0.001])
    previous_error = np.zeros(model.nv)
    integral_error = np.zeros(model.nv)
    desired_position = target_angles_values

    # breakpoint()
    for step in range(steps):
        for i, name in enumerate(joint_names):
            
            current_position = data.qpos[i]
            error = desired_position[i] - current_position
            
            integral_error[i] += error
            derivative = error - previous_error[i]
            
            # PID Control Signal
            control_signal = kp[i] * error + ki[i] * integral_error[i] + kv[i] * derivative
            data.ctrl[i] = control_signal  # Apply control
            
            previous_error[i] = error
            joint_angle_history[name].append(current_position)
        # Proceed with simulation step

        mujoco.mj_step(model, data)
        Viewer.sync()
        time.sleep(time_step)

    print("Simulation ended.")


fig, axs = plt.subplots(2, 4, figsize=(20, 10))  # Adjust subplot layout as needed
for idx, name in enumerate(joint_names):
    ax = axs[idx // 4, idx % 4]
    ax.plot(joint_angle_history[name])
    plt.title(f'Joint {name} Angle Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Angle (radians)')
    
    ax.set_xlim([0, len(joint_angle_history[name])])
    ax.set_ylim([min(joint_angle_history[name]) - 0.1, max(joint_angle_history[name]) + 0.1])
    
    plt.savefig(f'./pictures/joint_{name}_angle.png')

plt.savefig('./pictures/joint_angles.png')
plt.tight_layout()
plt.show()
