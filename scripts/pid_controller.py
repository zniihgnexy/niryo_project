import mujoco
"""
A proportional-integral-derivative (PID) controller class.
Args:
    Kp (float): Proportional gain.
    Ki (float): Integral gain.
    Kd (float): Derivative gain.
Attributes:
    Kp (float): Proportional gain.
    Ki (float): Integral gain.
    Kd (float): Derivative gain.
    integral_error (float): Integral error.
    prev_error (float): Previous error.
    dt (float): Time step.
Methods:
    calculate(target, current): Calculates the control output based on the target and current values.
"""
"""
A PID controller class with derivative filter.
Args:
    Kp (float): Proportional gain.
    Ki (float): Integral gain.
    Kd (float): Derivative gain.
    tau (float, optional): Time constant for the filter. Defaults to 0.1.
    dt (float, optional): Time step. Defaults to 0.005.
Attributes:
    pid (PIDController): PID controller object.
    tau (float): Time constant for the filter.
    dt (float): Time step.
    previous_derivative (float): Previous derivative value.
Methods:
    calculate(setpoint, measurement): Calculates the control output based on the setpoint and measurement values.
"""
"""
An advanced PID controller class with feedforward control.
Args:
    kp (float): Proportional gain.
    ki (float): Integral gain.
    kd (float): Derivative gain.
    setpoint (float): Setpoint value.
    ff_gain (float, optional): Feedforward gain. Defaults to 0.1.
Attributes:
    kp (float): Proportional gain.
    ki (float): Integral gain.
    kd (float): Derivative gain.
    setpoint (float): Setpoint value.
    ff_gain (float): Feedforward gain.
    integral (float): Integral value.
    last_error (float): Last error value.
Methods:
    update(measurement, delta_time): Updates the controller based on the measurement and time step.
    soft_start_control(current_position, target_position, step_size, max_increment): Calculates the control output for soft start control.
    compute_feedforward(target_position, dynamic_parameters): Computes the feedforward value based on the target position and dynamic parameters.
    control_with_feedforward(target_position, current_position): Calculates the control output with feedforward control.
"""
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
        self.dt = 0.02

    def calculate(self, target, current):
        error = target - current
        self.integral_error += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        output = (self.Kp * error) + (self.Ki * self.integral_error) + (self.Kd * derivative)
        self.prev_error = error
        return output


class PIDControllerWithDerivativeFilter:
    def __init__(self, Kp, Ki, Kd, tau=0.1, dt=0.005):
        self.pid = PIDController(Kp, Ki, Kd)
        self.tau = tau  # Time constant for the filter
        self.dt = dt
        self.previous_derivative = 0

    def calculate(self, setpoint, measurement):
        error = setpoint - measurement
        derivative = (error - self.pid.prev_error) / self.dt
        # Apply low-pass filter to the derivative term
        filtered_derivative = (self.tau * self.previous_derivative + self.dt * derivative) / (self.tau + self.dt)
        output = self.pid.Kp * error + self.pid.Ki * self.pid.integral_error + self.pid.Kd * filtered_derivative
        self.previous_derivative = filtered_derivative
        self.pid.prev_error = error
        return output

class AdvancedPIDController:
    def __init__(self, kp, ki, kd, setpoint, ff_gain=0.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.ff_gain = ff_gain
        self.integral = 0
        self.last_error = 0

    def update(self, measurement, delta_time):
        error = self.setpoint - measurement
        self.integral += error * delta_time
        derivative = (error - self.last_error) / delta_time
        self.last_error = error
        pid_output = (self.kp * error + self.ki * self.integral + self.kd * derivative)
        feedforward_output = self.ff_gain * self.setpoint
        return pid_output + feedforward_output
    
    def soft_start_control(self, current_position, target_position, step_size, max_increment):
        # Calculate incremental step
        direction = np.sign(target_position - current_position)
        increment = direction * min(step_size, abs(target_position - current_position), max_increment)
        return current_position + increment

    def compute_feedforward(self, target_position, dynamic_parameters):
        # Placeholder for a dynamic model or empirical data
        # For simplicity, just return a fraction of the target position
        return 0.1 * target_position

    def control_with_feedforward(self, target_position, current_position):
        feedforward_value = self.compute_feedforward(target_position, {})
        pid_output = self.update(current_position, 1)  # Assuming delta_time = 1 for simplicity
        return current_position + pid_output + feedforward_value