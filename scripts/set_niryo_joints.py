#!/usr/bin/env python

import math
import os
import mujoco
from xml.dom import minidom
from pyquaternion import Quaternion

class NiryoTwo:
    def __init__(self, model_xml_path):
        self.script_dir = os.path.dirname(__file__)
        self.model_xml_path = model_xml_path
        self.model = None
        self.physics = None

        # Load the model
        self.load_model()

    def load_model(self):
        print("Loading model from: ", self.model_xml_path)
        with open(self.model_xml_path, 'r') as xml_file:
            xml_string = xml_file.read()
            # print("xml_string: ", xml_string)
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        self.physics = mujoco.MjData(self.model)

    def apply_joint_angles(self, joint_angles):
        # joint_angles should be a list of angles in degrees
        assert len(joint_angles) == 6, "There must be 6 joint angles."

        for i, angle in enumerate(joint_angles):
            rad_angle = math.radians(angle)
            self.physics.qpos[i] = rad_angle

        mujoco.mj_forward(self.model, self.physics)

    def simulate(self):
        # Function to visualize the simulation
        renderer = mujoco.MujocoRenderer(self.model)
        while True:
            mujoco.mj_step(self.model, self.physics)
            renderer.render(self.physics)

    def update_robot_model(self, joint_angles):
        # Update the XML file with new joint angles
        doc = minidom.parse(self.model_xml_path)
        joints = doc.getElementsByTagName("joint")

        for i, joint in enumerate(joints):
            if i < len(joint_angles):  # Only update known joints
                quat = Quaternion(axis=[0, 0, 1], angle=math.radians(joint_angles[i]))
                quat_str = "{} {} {} {}".format(quat[0], quat[1], quat[2], quat[3])
                joint.setAttribute('quat', quat_str)

        with open(self.model_xml_path, 'w') as file:
            doc.writexml(file)

        # Reload model to apply changes
        self.load_model()

if __name__ == "__main__":
    model_path = "/home/xz2723/niryo_project/meshes/niryo_arm_table.xml"
    print("Model path: ", model_path)
    print("current path: ", os.getcwd())
    niryo_robot = NiryoTwo(model_path)

    # Set initial joint angles
    initial_angles = [10, -20, 30, -40, 50, -60]
    niryo_robot.apply_joint_angles(initial_angles)

    # Simulate
    niryo_robot.simulate()
