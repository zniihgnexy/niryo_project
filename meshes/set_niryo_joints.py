import os
import math
from xml.dom import minidom
from pyquaternion import Quaternion
import mujoco
import mujoco.viewer

class NiryoTwo:
    def __init__(self, model_xml_path):
        self.script_dir = os.path.dirname(os.path.realpath(__file__))  # Correctly obtaining the script directory
        self.model_xml_path = os.path.join(self.script_dir, model_xml_path)
        self.model = None
        self.physics = None

        # Load the model
        self.load_model()

    def load_model(self):
        print("Loading model from: ", self.model_xml_path)
        self.model = mujoco.MjModel.from_xml_path(self.model_xml_path)  # Changed to use from_xml_path
        self.physics = mujoco.MjData(self.model)
        print("type of physics: ", type(self.physics))

    def apply_joint_angles(self, joint_angles):
        assert len(joint_angles) == 6, "There must be 6 joint angles."
        for i, angle in enumerate(joint_angles):
            rad_angle = math.radians(angle)
            self.physics.qpos[i] = rad_angle
        mujoco.mj_forward(self.model, self.physics)

    def simulate(self):
        # Ensure correct visualization setup
        renderer = mujoco.Renderer(self.model)
        while True:
            mujoco.mj_step(self.model, self.physics)
            renderer.render()

    def update_robot_model(self, joint_angles):
        doc = minidom.parse(self.model_xml_path)
        joints = doc.getElementsByTagName("joint")
        for i, joint in enumerate(joints):
            if i < len(joint_angles):
                quat = Quaternion(axis=[0, 0, 1], angle=math.radians(joint_angles[i]))
                quat_str = "{} {} {} {}".format(*quat.elements)
                joint.setAttribute('quat', quat_str)

        with open(self.model_xml_path, 'w') as file:
            doc.writexml(file)

        self.load_model()

def simulation(self, steps=1):
    for _ in range(steps):
        mujoco.mj_step(niryo_robot.model, niryo_robot.physics)
        renderer.render()

if __name__ == "__main__":
    model_path = "niryo_arm.xml"
    niryo_robot = NiryoTwo(model_path)
    
    # viewer = mujoco.viewer.MujocoViewer(niryo_robot.model, niryo_robot.physics)

    with mujoco.viewer.launch(niryo_robot.model, niryo_robot.physics) as viewer:
        mujoco.mj_step(niryo_robot.model, niryo_robot.physics)
        viewer.render()

        while True:
            # Apply a set of joint angles
            niryo_robot.apply_joint_angles([10, 10, 10, 10, 10, 10])
            # Simulate for a short period to reflect the changes
            for _ in range(100):  # simulate 100 steps at a time
                mujoco.mj_step(niryo_robot.model, niryo_robot.physics)
                viewer.render()

            niryo_robot.apply_joint_angles([20, 20, 20, 20, 20, 20])
            # Simulate for another short period
            for _ in range(100):
                mujoco.mj_step(niryo_robot.model, niryo_robot.physics)
                viewer.render()

