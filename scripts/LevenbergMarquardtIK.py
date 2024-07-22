import mujoco
import numpy as np

class LevenbergMarquardtIK:
    def __init__(self, model, data, step_size, tol, alpha, damping, movable_joints_indices):
        self.model = model
        self.data = data
        self.step_size = step_size
        self.tol = tol
        self.alpha = alpha
        self.damping = damping
        self.movable_joints_indices = movable_joints_indices
    
    def check_joint_limits(self, q):
        """Check if the joints are under or above their limits"""
        for i in range(len(q)):
            q[i] = max(self.model.jnt_range[self.movable_joints_indices[i]][0], 
                       min(q[i], self.model.jnt_range[self.movable_joints_indices[i]][1]))

    def calculate(self, goal, init_q, body_id):
        """Calculate the desired joint angles for the goal"""
        self.data.qpos[self.movable_joints_indices] = init_q
        mujoco.mj_forward(self.model, self.data)
        current_pose = self.data.body(body_id).xpos
        error = np.subtract(goal, current_pose)

        while (np.linalg.norm(error) >= self.tol):
            # Calculate Jacobian
            jacp = np.zeros((3, self.model.nv))  # Translational Jacobian
            jacr = np.zeros((3, self.model.nv))  # Rotational Jacobian
            mujoco.mj_jac(self.model, self.data, jacp, jacr, current_pose, body_id)
            
            # Extract only the columns corresponding to the movable joints
            jacp_movable = jacp[:, self.movable_joints_indices]
            
            # Calculate delta of joint q
            n = jacp_movable.shape[1]
            I = np.identity(n)
            product = jacp_movable.T @ jacp_movable + self.damping * I
            
            if np.isclose(np.linalg.det(product), 0):
                j_inv = np.linalg.pinv(product) @ jacp_movable.T
            else:
                j_inv = np.linalg.inv(product) @ jacp_movable.T
            
            delta_q = j_inv @ error
            
            # Compute next step
            self.data.qpos[self.movable_joints_indices] += self.step_size * delta_q
            
            # Check joint limits
            self.check_joint_limits(self.data.qpos[self.movable_joints_indices])
            
            # Compute forward kinematics
            mujoco.mj_forward(self.model, self.data) 
            
            # Calculate new error
            current_pose = self.data.body(body_id).xpos
            error = np.subtract(goal, current_pose)
            
        return self.data.qpos[self.movable_joints_indices].copy()
