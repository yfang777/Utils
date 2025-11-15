from xarm import version
from xarm.wrapper import XArmAPI
import numpy as np

interface="192.168.1.224"
ArmAPI = XArmAPI(interface, baud_checkset=False)
init_pose=[256.7, 5.1, 400.1, 178.9, 0.0, 1.4]

init_servo_angle = [0.0, -45.0, 0.0, 30.0, 0.0, 75.0, 0.0]

# init_servo_angle = [-0.1384, -0.1688, -0.3374,  0.5164, -0.0167,  0.7233, -0.0278]

ArmAPI.clean_warn()
ArmAPI._arm.clean_error()
ArmAPI._arm.motion_enable(True)
ArmAPI._arm.set_mode(0) # Position control mode
ArmAPI._arm.set_state(0) # Set to ready state

gripper_enable = True
if gripper_enable:
    ArmAPI.set_gripper_enable(True)
    ArmAPI.set_gripper_mode(0)
    ArmAPI.clean_gripper_error()


# Get the current angles of all 7 joints
# The get_servo_angle() function returns a tuple (code, angles_list)
code, angles = ArmAPI.get_servo_angle()

if code == 0:
    print("Current servo angles (in degrees):")
    print(angles)
else:
    print(f"Failed to get angles. Error code: {code}")

# reset
# ArmAPI.set_servo_angle(angle=init_servo_angle, isradian=True, wait=True)
ArmAPI.set_servo_angle(angle=init_servo_angle, isradian=False, wait=True)

code, angles = ArmAPI.get_servo_angle()
code, pos = ArmAPI.get_position(is_radian=True)

if code == 0:
    print("angles:", angles)
    print("pos:", pos)
else:
    print(f"Failed to get angles. Error code: {code}")

second_servo_angle = [-10.2, -24.3, -11.3, 29.6, -6.3, 53.2, -17.1]
ArmAPI.set_servo_angle(angle=second_servo_angle, isradian=False, wait=True)

import time
time.sleep(3)
