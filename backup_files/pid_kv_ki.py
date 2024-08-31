import mujoco
"""
This script performs a PID parameter optimization for a specific joint in a Mujoco model. It tests different combinations of Kv (velocity gain) and Ki (integral gain) values to find the best PID parameters that result in the lowest shift from the initial position.

Parameters:
    None

Returns:
    None

Raises:
    None
"""
import numpy as np
import time
from mujoco import viewer
import matplotlib.pyplot as plt

# Load the model
model_path = '/home/xz2723/niryo_project/meshes/mjmodel.xml'
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

initialize_angles = np.array([0.5000, 0.5000, 0.5000, 0.0000, 0.0000, 0.0000, 0.00000, 0.00000])

# Define joint names and target angles
joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'left_clamp_joint', 'right_clamp_joint']
target_angles = np.array([0.5000, 0.5000, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000])
joint_indices = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names}

joint_angle_history = {name: [] for name in joint_names}

# Select the joint to test
joint_to_test = 'joint_6'
index_to_test = joint_names.index(joint_to_test)

# Initialize arrays for storing PID parameters and histories
kv_values = np.linspace(28, 38, 10)   # Example Kv range
ki_values = np.linspace(0, 0.1, 10)    # Example Ki range

results = []
shifts = []

for name in joint_names:
    joint_id = joint_indices[name]
    data.qpos[joint_id] = initialize_angles[joint_indices[name]]
    joint_angle_history[name].append(data.qpos[joint_id])

# Initialize the viewer
with viewer.launch_passive(model, data) as Viewer:
    Viewer.sync()
    # Hold initial position
    initial_hold_time = 1  # seconds
    start_time = time.time()
    print("Holding initial position for 2 seconds...")
    while time.time() - start_time < initial_hold_time:
        Viewer.sync()
        time.sleep(0.05)
    print("Initial position hold complete.")

    kp_test = np.array([179, 180, 180, 180, 173, 173, 0.25, 0.25])
    for kv_test in kv_values:
        for ki_test in ki_values:
            # Reset the integral and previous error at the start of each PID combination test
            integral_error = 0
            previous_error = 0

            print(f"Testing PID: Kp={kp_test[index_to_test]}, Kv={kv_test}, Ki={ki_test} for {joint_to_test}")

            # Run simulation for each combination
            temp_angle_history = []
            for step in range(100):  # Number of steps to simulate
                current_position = data.qpos[index_to_test]
                error = target_angles[index_to_test] - current_position

                integral_error += error
                derivative = error - previous_error

                # Manual PID calculation
                control_signal = kp_test[index_to_test] * error + ki_test * integral_error + kv_test * derivative
                data.ctrl[index_to_test] = control_signal

                previous_error = error
                temp_angle_history.append(current_position)

                mujoco.mj_step(model, data)
                Viewer.sync()
                time.sleep(0.01)  # Simulation time step

            # Assess stability or performance here by analyzing temp_angle_history
            max_shift = np.max(np.abs(temp_angle_history - initialize_angles[index_to_test]))
            results.append((kp_test, kv_test, ki_test, max_shift))
            shifts.append(max_shift)
            # print("max shifts for this set is: ", max_shift)

    # Evaluate results to determine the best PID parameters
    # Sorting results by max_shift to find the PID parameters with the lowest shift (indicating stability)
    results.sort(key=lambda x: x[3])
    best_parameters = results[0]
    print(f"Best PID parameters: Kp={kp_test[index_to_test]}, Kv={best_parameters[1]}, Ki={best_parameters[2]} with shift={best_parameters[3]}")

# plot the shift and the corresponding Kv and Ki values
best_kv = best_parameters[1]
best_ki = best_parameters[2]

plt.figure()
plt.plot(shifts)
plt.xlabel('Index')
plt.ylabel('Max Shift')
plt.title('Max Shift vs. Index')
plt.savefig('./pictures/kvki_shift_index' + best_kv + '_' + best_ki + 'for_joint' + joint_to_test +'.png')