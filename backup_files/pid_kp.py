import numpy as np
"""
PID Controller Class and Joint Testing
This code defines a PIDController class and performs joint testing using the Mujoco physics engine. The PIDController class implements a proportional-integral-derivative controller with adjustable gains (Kp, Ki, Kd). The joint testing involves finding the best combination of Kp, Ki, and Kd values for each joint to achieve smooth and accurate movement.
Classes:
- PIDController: Implements a PID controller with adjustable gains.
Functions:
- calculate: Calculates the control signal based on the target and current values.
Variables:
- model_path: The path to the Mujoco model XML file.
- joint_names: A list of joint names to test.
- test_names: A list of joint names to test.
- joint_ranges_list: A dictionary of joint names and their corresponding target ranges.
- joint_ranges: A dictionary of joint names and their corresponding target ranges for testing.
- kp_values: An array of Kp values to test.
- kd_values: An array of Kd values to test.
- ki_values: An array of Ki values to test.
- joint_indices: A dictionary of joint names and their corresponding indices in the Mujoco model.
- pids: A dictionary of joint names and their corresponding PIDController objects.
- best_kp_results: A dictionary of joint names and their best Kp values.
- best_kd_results: A dictionary of joint names and their best Kd values.
- best_ki_results: A dictionary of joint names and their best Ki values.
- oscillation_history: A dictionary of Kp values and their corresponding oscillation values.
- fixed_positions: A dictionary of joint names and their fixed positions for testing.
"""
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

joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'left_clamp_joint', 'right_clamp_joint']
test_names = ['joint_3']  # Specify the joints to test

joint_ranges_list = {
    'joint_1': np.linspace(2.000, -2.000, 3),
    'joint_2': np.linspace(0.5, -0.8, 3),
    'joint_3': np.linspace(-1.34, 1.57, 3),
    'joint_4': np.linspace(-2.089, 2.089, 10),
    'joint_5': np.linspace(-1.74533, 1.91986, int(10)),
    'joint_6': np.linspace(-2.57436, 2.57436, 3),
    'left_clamp_joint': np.linspace(0, -0.012,3),
    'right_clamp_joint': np.linspace(0.012, 0, 3)
}

# joint_ranges = {'joint_2': np.linspace(-1.5708, 0.640187, 10)}
joint_ranges = {name: joint_ranges_list[name] for name in test_names}

# breakpoint()
kp_values = np.linspace(100, 200, 5)
kd_values = np.linspace(0, 100, 5)
ki_values = np.linspace(0, 1, 2)

joint_indices = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names}
# pids = {name: PIDController(180, 0.06, 33) for name in joint_names}

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

best_kp_results = {}
best_kd_results = {}
best_ki_results = {}

oscillation_history = {}

fixed_positions = {
    'joint_1': 0.00369,
    'joint_2': 0.61,
    'joint_3': -0.511,
    'joint_4': -1.48,
    'joint_5': -1.49,
    'joint_6': 0.000,
    'left_clamp_joint': -0.004,
    'right_clamp_joint': 0.00036
}

'''
fixed position for different joints using for the tests of Kp values

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

situations:
joint_2: -0.03 and after will simply colapse on the table
joint_3: range from 0.46 to -0.51, can't reach 1.5 or maybe can reach but with the joint_2 not in the upright position
joint_4: testing for joint_4 is ok but joint 3 angle value can automatically change to -0.011 or around
'''

with viewer.launch_passive(model, data) as Viewer:
    Viewer.sync()
    # Initialize all joint positions
    # for name in joint_names:
    #     joint_id = joint_indices[name]
    #     data.qpos[joint_id] = fixed_positions[name]
    #     Viewer.sync()
    #     time.sleep(0.5) 

    # mujoco.mj_step(model, data)
    # Viewer.sync()
    # time.sleep(0.5)

    for name in test_names:
        best_smoothness_score = float('inf')
        best_kp = None
        best_kd = None
        best_ki = None

        for Kp in kp_values:
            # Reset the joint under test to its starting position before each Kp test
            data.qpos[joint_indices[name]] = fixed_positions[name]
            mujoco.mj_step(model, data)
            Viewer.sync()
            time.sleep(0.5)
            
            for Kd in kd_values:
                
                for Ki in ki_values:
                    pids[name].Kp = Kp
                    pids[name].Kd = Kd
                    pids[name].Ki = Ki
                    position_history = []
                    for target in joint_ranges[name]:
                        print(f"Testing {name} at Kp = {Kp} Kd = {Kd}  Ki = {Ki} with target = {target}")

                        # Reset other joints to their fixed positions for each target
                        for init_name in joint_names:
                            if init_name != name:
                                data.qpos[joint_indices[init_name]] = fixed_positions[init_name]

                        mujoco.mj_step(model, data)
                        Viewer.sync()
                        time.sleep(0.5)  # Allow time for the system to stabilize at the start position

                        start_time = time.time()
                        while time.time() - start_time < 5:
                            
                            current_position = data.qpos[joint_indices[name]]
                            # control_signal = pids[name].calculate(target, current_position)
                            # data.ctrl[joint_indices[name]] = control_signal
                            # mujoco.mj_step(model, data)
                            
                            for ori_name in joint_names:
                                if ori_name != name:
                                    control_signal = pids[name].calculate(target, fixed_positions[name])
                                    data.ctrl[joint_indices[name]] = control_signal
                                    mujoco.mj_step(model, data)
                                    Viewer.sync()
                                    time.sleep(0.01)
                                else:
                                    control_signal = pids[name].calculate(target, current_position)
                                    data.ctrl[joint_indices[name]] = control_signal
                                    mujoco.mj_step(model, data)
                                    Viewer.sync()
                                    time.sleep(0.01)

                            Viewer.sync()
                            time.sleep(0.01)
                            position_history.append(current_position)
                        
                        # get the last second joint angle values
                        last_second_joint_angle = position_history[-50:]
                        # get teh mean value
                        mean_value = np.mean(last_second_joint_angle)
                        print("last second joint angle values", mean_value)

                        # Evaluate smoothness
                        mean_value = np.mean(position_history)
                        overshoots = np.max(np.abs(np.array(position_history) - target)) - target
                        steadiness = np.var(position_history[-50:])  # Variance of the last 50 samples
                        difference = np.abs(mean_value - target)
                        smoothness_score = overshoots + steadiness + difference
                        
                        # check oscillation
                        oscillation = np.abs(np.max(position_history) - np.min(position_history))
                        print("oscillation", oscillation)
                        # save teh oscaillation value and the Kp value in one dict
                        oscillation_history[Kp] = oscillation

                        if smoothness_score < best_smoothness_score:
                            best_smoothness_score = smoothness_score
                            best_kp = Kp
                            best_kd = Kd
                            best_ki = Ki
                        
                print(f"Test completed for {name} at Kp = {Kp}  Ki = {Ki}  Kd = {Kd} with smoothness score = {best_smoothness_score}")
        best_kp_results[name] = best_kp
        best_kd_results[name] = best_kd
        best_ki_results[name] = best_ki
        print(f"Best Kp for {name} is {best_kp} and best Kd is {best_kd} and best Ki is {best_ki} with smoothness score {best_smoothness_score}")
        print("oscillation history", oscillation_history)

    print("Best Kp values for tested joints:", best_kp_results)
    print("Best Kd values for tested joints:", best_kd_results)
    print("Best Ki values for tested joints:", best_ki_results)
