import mujoco
import numpy as np

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