import numpy as np
import mujoco
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
        self.dt = 0.01  # Control period, match simulation step

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

# joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
joint_names = ['joint_1']
fixed_positions = {
    'joint_1': -3.05433,
    'joint_2': -0.785,
    'joint_3': 0.785,
    'joint_4': 0.0,
    'joint_5': -0.785,
    'joint_6': 0.785,
    'left_clamp_joint': -0.012,
    'right_clamp_joint': 0.012
}

# joint_ranges = {name: np.linspace(-3.05433, 3.05433, 20) for name in joint_names}
joint_ranges = {
    'joint_1': np.linspace(-3.05433, 3.05433, 10),
    'joint_2': np.linspace(-1.5708, 0.640187, 10),
    'joint_3': np.linspace(-1.39749, 1.5708, 10),
    'joint_4': np.linspace(-3.05433, 3.05433, 10),
    'joint_5': np.linspace(-1.74533, 1.91986, 10),
    'joint_6': np.linspace(-2.57436, 2.57436, 10),
    'left_clamp_joint': np.linspace(-0.012, 0, 10),  # Assuming fixed if needed
    'right_clamp_joint': np.linspace(0, 0.012, 10)
}

max_range_value = {name: max(abs(joint_ranges[name][0]), abs(joint_ranges[name][-1])) for name in joint_names}

# breakpoint()
kp_values = np.linspace(150, 250, 5)
joint_indices = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names}
pids = [PIDController(180, 0.06, 33) for _ in range(len(joint_names))]

best_kp_results = {}

# #initialize the jopints
# for name in joint_names:
#     data.qpos[joint_indices[name]] = fixed_positions[name]

with viewer.launch_passive(model, data) as Viewer:
    Viewer.sync()

    # Initialize the joint positions
    for name in joint_names:
        joint_id = joint_indices[name]
        data.qpos[joint_id] = fixed_positions[name]
        Viewer.sync()
        time.sleep(0.5)  # Allow time for the joint to settle at the start position

    for name in joint_names:
        best_error = float('inf')
        best_kp = None

        for Kp in kp_values:
            pids[joint_indices[name]].Kp = Kp
            total_error = 0
            
            if data.qpos[joint_indices[name]] != fixed_positions[name]:
                data.qpos[joint_indices[name]] = fixed_positions[name]
                mujoco.mj_step(model, data)
                Viewer.sync()
                time.sleep(0.5)
            
            else:
                print(f"Joint {name} is already at the fixed position")

            for target in joint_ranges[name]:
                print(f"Testing {name} at Kp = {Kp} towards target {target}")
                start_time = time.time()

                while time.time() - start_time < 2:  # Control for 2 seconds
                    current_position = data.qpos[joint_indices[name]]
                    error = abs(target - current_position)
                    total_error += error
                    control_signal = pids[joint_indices[name]].calculate(target, current_position)
                    data.ctrl[joint_indices[name]] = control_signal
                    mujoco.mj_step(model, data)
                    Viewer.sync()
                    time.sleep(0.01)  # Time step between control updates

            # Reset to the fixed position between trials only when the angle exceeds the range
            if data.qpos[joint_indices[name]] >= max_range_value[name]:
                data.qpos[joint_indices[name]] = fixed_positions[name]
                mujoco.mj_step(model, data)
                Viewer.sync()
                time.sleep(0.5)  # Allow time for the joint to settle back at the fixed position

            if total_error < best_error:
                best_error = total_error
                best_kp = Kp

            print(f"Test completed for {name} at Kp = {Kp} with total error = {total_error}")

        best_kp_results[name] = best_kp
        print(f"Best Kp for {name} is {best_kp} with error {best_error}")

print("Best Kp values for all joints:", best_kp_results)