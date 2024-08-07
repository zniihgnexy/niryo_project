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
        self.dt = 0.01

    def calculate(self, target, current):
        error = target - current
        self.integral_error += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        output = (self.Kp * error) + (self.Ki * self.integral_error) + (self.Kd * derivative)
        self.prev_error = error
        return output

# Load model and setup
model_path = '/home/xz2723/niryo_project/meshes/mjmodel.xml'
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
test_names = ['joint_2']  # Specify the joints to test
fixed_positions = {
    'joint_1': -0.785,
    'joint_2': -1.4000,
    'joint_3': 1.2000,
    'joint_4': 0.0,
    'joint_5': -0.785,
    'joint_6': 0.785,
    'left_clamp_joint': -0.012,
    'right_clamp_joint': 0.012
}

joint_ranges_list = {
    'joint_1': np.linspace(-3.05433, 3.05433, 10),
    'joint_2': np.linspace(-1.4000, 0.640187, 10),
    'joint_3': np.linspace(-1.39749, 1.5708, 10),
    'joint_4': np.linspace(-3.05433, 3.05433, 10),
    'joint_5': np.linspace(-1.74533, 1.91986, 10),
    'joint_6': np.linspace(-2.57436, 2.57436, 10),
    'left_clamp_joint': np.linspace(-0.012, 0, 10),  # Assuming fixed if needed
    'right_clamp_joint': np.linspace(0, 0.012, 10)
}

# joint_ranges = {'joint_2': np.linspace(-1.5708, 0.640187, 10)}
joint_ranges = {name: joint_ranges_list[name] for name in test_names}
kp_values = np.linspace(200, 250, 2)
joint_indices = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names}
# pids = {name: PIDController(180, 0.06, 33) for name in joint_names}

pids = {
    'joint_1': PIDController(180, 0.06, 33),
    'joint_2': PIDController(180, 0.06, 33),
    'joint_3': PIDController(180, 0.06, 33),
    'joint_4': PIDController(180, 0.06, 33),
    'joint_5': PIDController(180, 0.06, 33),
    'joint_6': PIDController(180, 0.06, 33),
    'left_clamp_joint': PIDController(180, 0.06, 33),
    'right_clamp_joint': PIDController(180, 0.06, 33)
}

best_kp_results = {}

with viewer.launch_passive(model, data) as Viewer:
    Viewer.sync()

    # Initialize all joint positions
    for name in joint_names:
        joint_id = joint_indices[name]
        data.qpos[joint_id] = fixed_positions[name]
        Viewer.sync()
        time.sleep(0.5) 

    mujoco.mj_step(model, data)
    Viewer.sync()
    time.sleep(0.5)

    for name in test_names:
        best_smoothness_score = float('inf')
        best_kp = None

        for Kp in kp_values:
            pids[name].Kp = Kp
            position_history = []
            
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

                while time.time() - start_time < 2:
                    current_position = data.qpos[joint_indices[name]]
                    control_signal = pids[name].calculate(target, current_position)
                    data.ctrl[joint_indices[name]] = control_signal
                    
                    mujoco.mj_step(model, data)
                    Viewer.sync()
                    time.sleep(0.01)
                    position_history.append(current_position)

                # Evaluate smoothness
                overshoots = np.max(np.abs(np.array(position_history) - target)) - target
                steadiness = np.var(position_history[-50:])  # Variance of the last 50 samples
                smoothness_score = overshoots + steadiness
                print(f"Smoothness score: {smoothness_score}")

            if smoothness_score < best_smoothness_score:
                best_smoothness_score = smoothness_score
                best_kp = Kp

            print(f"Test completed for {name} at Kp = {Kp} with smoothness score = {best_smoothness_score}")

        best_kp_results[name] = best_kp
        print(f"Best Kp for {name} is {best_kp} with smoothness score {best_smoothness_score}")

print("Best Kp values for tested joints:", best_kp_results)