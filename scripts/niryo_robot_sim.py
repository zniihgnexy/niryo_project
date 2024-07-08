from pyniryo2 import *

robot = NiryoRobot("127.0.0.1")

# Connect to robot & calibrate
# robot = NiryoRobot(robot_ip_address)
robot.arm.calibrate_auto()
# Move joints
robot.arm.move_joints([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# Turn learning mode ON
robot.arm.set_learning_mode(True)
# Stop TCP connection
robot.end()