import mujoco
import numpy as np
import time
from mujoco import viewer
import matplotlib.pyplot as plt
from pid_controller import PIDController
from kinemetic import robot_chain

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
    'joint_5': -1.49,
    'joint_6': 0.901,
    'left_clamp_joint': -0.004,
    'right_clamp_joint': 0.00036
}

initialize_angles = np.array([fixed_positions[name] for name in joint_names])

target_position = [0.1, 0.2, 0.05]  # Specify the 3D position

send_in_position = target_position
target_angles = robot_chain.inverse_kinematics(send_in_position)
print("Target angles:", target_angles)

check_positions = robot_chain.forward_kinematics(target_angles)
print("Check angles:", check_positions)

# breakpoint()

control_target_angles = target_angles[1:]
print("Control target angles:", control_target_angles)

# add two more angles to teh control target angles
control_target_angles = np.append(control_target_angles, [0.000, 0.000])

print("Control target angles:", control_target_angles)

# breakpoint()

# PID controller setup
# kp = np.array([179, 180, 180, 180, 173, 173, 0.25, 0.25])
# kv = np.array([33, 33, 33, 33, 50, 50, 1, 1])
# ki = np.array([0, 0, 0, 0, 0, 0, 0, 0])
# pids = [PIDController(kp[i], ki[i], kv[i]) for i in range(len(kp))]

pids = {
    'joint_1': PIDController(100, 0.5, 100),
    'joint_2': PIDController(100, 0.8, 50),
    'joint_3': PIDController(100, 0.06, 80),
    'joint_4': PIDController(100, 0.06, 80),
    'joint_5': PIDController(100, 0.06, 80),
    'joint_6': PIDController(162.5, 0.06, 33),
    'left_clamp_joint': PIDController(10, 0.06, 80),
    'right_clamp_joint': PIDController(10, 0.06, 80)
}

joint_angle_history = {name: [] for name in joint_names}

# Function to control arm position
def control_arm_to_position(model, data, joint_names, joint_indices, pids, target_angles):
    for name in joint_names:
        joint_index = joint_indices[name]
        current_position = data.qpos[joint_index]
        target_angle = target_angles[joint_indices[name]]
        control_signal = pids[name].calculate(target_angle, current_position)
        data.ctrl[joint_index] = control_signal
        joint_angle_history[name].append(current_position)


# Initialize the passive viewer
with viewer.launch_passive(model, data) as Viewer:
    Viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 0
    Viewer.sync()

    # Simulation loop for synchronized movement
    duration = 20  # seconds
    steps = int(duration * 50)  # Assuming 50 Hz simulation frequency
    tolerance = 0.01  # tolerance for joint position
    
    # Set initial joint positions
    for name, angle in fixed_positions.items():
        joint_index = joint_indices[name]
        data.qpos[joint_index] = angle
        print("initial angles:", joint_index, angle)
        Viewer.sync()
        time.sleep(0.02)
    
    for _ in range(steps):
        # print("Moving Joints 3, 4, and 5 first...")
        control_arm_to_position(model, data, ['joint_3', 'joint_4', 'joint_5'], joint_indices, pids, control_target_angles)
        mujoco.mj_step(model, data)
        Viewer.sync()
        time.sleep(0.02)
        
        # print("the rest of the joints...")
        control_arm_to_position(model, data, ['joint_1', 'joint_2', 'joint_6', 'left_clamp_joint', 'right_clamp_joint'], joint_indices, pids, control_target_angles)
        mujoco.mj_step(model, data)
        Viewer.sync()
        time.sleep(0.02)

    print("All joints have moved towards their target positions.")


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

for name in joint_names:
    index = joint_indices[name]
    # print(f"Joint {data.geom(index).name} position: {data.geom(index).xpos}")

# Capture the final end-effector position
end_position = [data.geom(6).xpos, data.geom(7).xpos]
print("Final end-effector position:", end_position)

end_angles = [data.qpos[joint_indices[name]] for name in joint_names]
print("Final joint angles:", end_angles)
