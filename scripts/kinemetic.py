from ikpy.chain import Chain
from ikpy.link import URDFLink

# # plot the robot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# loadlink from urdf file
urdf_file_path = "niryo_robot_old.urdf"
robot_chain = Chain.from_urdf_file(urdf_file_path)
