import mujoco
import numpy as np
import time
from mujoco import viewer
import matplotlib.pyplot as plt
from pid_controller import PIDController
from scripts.kinemetic_test import robot_chain

# Load the model
model_path = '/home/xz2723/niryo_project/meshes/mjmodel.xml'
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Joint names and target angles
initialize_angles = np.array([1.0000, 0.5000, 0.5000, 0.0000, 0.0000, 0.0000, 0.00000, 0.00000])

joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'left_clamp_joint', 'right_clamp_joint']
joint_angle_history = {name: [] for name in joint_names}
joint_indices = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names}

# Initialize joint positions according to target angles
for name in joint_names:
    joint_id = joint_indices[name]
    data.qpos[joint_id] = initialize_angles[joint_indices[name]]
    joint_angle_history[name].append(data.qpos[joint_id])
    
# Initialize kinematic solver
# kinematic = Kinematic("./niryo_two.urdf")
target_position = [0.3, 0.3, 0.3]  # Modify this as needed
target_angles = robot_chain.inverse_kinematics(target_position)
# target_angles = kinematic.inverse_kinematics(target_position)


target_angles = np.array([1.0000, 0.5000, -1.0000, 0.0000, 0.0000, 0.0000, 0.00000, 0.00000])
print("Inverse Kinematics Result:", target_angles)

# Convert target angles to a numpy array if it's not empty or None
for i in range(len(target_angles)):
    if target_angles[i] is None:
        target_angles[i] = 0
    else:
        # target_angles[i] = round(target_angles[i], 8)
        # don't omit any zeros after the decimal point
        target_angles[i] = round(target_angles[i], 9) if target_angles[i] != 0 else 0

print("Processed Target Angles:", target_angles)
print("first target angle:", target_angles[0])

# breakpoint()
# if target_angles:
#     target_angles = np.array([round(angle, 4) for angle in target_angles])
# else:
#     print("No valid target angles calculated.")
#     target_angles = np.zeros(len(joint_names))  # Default to zero if no angles calculated

# print("Processed Target Angles:", target_angles)

# Map joint names to indices
joint_indices = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names}

# Initialize the passive viewer
with viewer.launch_passive(model, data) as Viewer:
    # Set viewer settings, such as enabling wireframe mode
    Viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 1
    Viewer.sync()

    # Hold initial position
    initial_hold_time = 2  # seconds
    start_time = time.time()
    print("Holding initial position for 2 seconds...")
    while time.time() - start_time < initial_hold_time:
        Viewer.sync()
        time.sleep(0.05)
    print("Initial position hold complete.")

    # PID controller setup
    kp = np.array([179, 180, 180, 180, 173, 173, 0.25, 0.25])
    kv = np.array([33, 100, 33, 33, 20, 20, 1, 1])
    ki = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    pids = [PIDController(kp[i], ki[i], kv[i]) for i in range(len(kp))]
    
    # Simulation parameters
    duration = 10  # seconds
    steps = int(duration * 50)  # Assuming 50 Hz simulation frequency
    time_step = 1 / 50
    # one_movement_time = 2  # seconds
    # trajectory = np.linspace(initialize_angles, target_angles, one_movement_time * 50)

    # Simulation loop
    for step in range(steps):
        for i, name in enumerate(joint_names):
            joint_id = joint_indices[name]
            current_position = data.qpos[joint_id]
            current_target_angles = target_angles[i]
            
            # if step < one_movement_time * 50:
            #     current_target_angles = trajectory[step][i]
            #     print("current_target_angles: ", current_target_angles)
            #     control_signal = pids[i].calculate(current_target_angles, current_position)
            # else:
            #     current_target_angles = target_angles[i]
            #     control_signal = pids[i].calculate(current_target_angles, current_position)
                

            error = target_angles[joint_id] - current_position

            # Calculate control signal using PID
            control_signal = pids[i].calculate(current_target_angles, current_position)
            data.ctrl[joint_id] = control_signal
            # print(f"Joint {name} at angle value {data.qpos[joint_indices[name]]} with target {current_target_angles} and control signal {control_signal}")
            joint_angle_history[name].append(current_position)

        mujoco.mj_step(model, data)
        Viewer.sync()
        time.sleep(time_step)

    print("Simulation ended.")
    print("final joint position: ", data.geom(6).name, data.geom(6).xpos, data.qpos[joint_indices['left_clamp_joint']])
    print("final joint angles: ", [data.qpos[joint_indices[name]] for name in joint_names])

# Plot joint angles over time
fig, axs = plt.subplots(2, 4, figsize=(20, 10))
for idx, name in enumerate(joint_names):
    ax = axs[idx // 4, idx % 4]
    ax.plot(joint_angle_history[name])
    ax.set_title(f'Joint {name} Angle Over Time')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Angle (radians)')
    ax.set_xlim([0, len(joint_angle_history[name])])
    ax.set_ylim([min(joint_angle_history[name]) - 0.1, max(joint_angle_history[name]) + 0.1])

plt.tight_layout()
plt.savefig('./pictures/joint_angles.png')
plt.show()
