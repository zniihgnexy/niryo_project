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

