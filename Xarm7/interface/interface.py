import os
import cv2
import time
import yaml
import pickle
import datetime
import numpy as np
from pathlib import Path

import pyrealsense2 as rs

import transform_utils as T
from ik_solver_sapien import KinHelper

from xarm.wrapper import XArmAPI


def angle2radian(angle):
    arr = np.array(angle, dtype=np.float32)
    return np.deg2rad(arr)

def radian2angle(radian):
    angle = np.array(radian, dtype=np.float32)
    return np.rad2deg(angle)
    
def convert(gripper_norm):
    return (gripper_norm - 0.5) / 2 * (-510) + 500 

class RealEnv():

    def __init__(self, ip: str = "192.168.1.224", rs_id: str = "239222300412"):
  
        self.dt = 1e-4
        self.act_freq = 30
        self.sim_2_act_time_factor = int(1 / self.dt / self.act_freq)
        self.render_mujoco = True
        self.render_width = 640
        self.render_height = 480
        self.act_freq = 30
        self.viewer = None
        self.init_servo_angle = np.array([0.0, -45.0, 0.0, 30.0, 0.0, 75.0, 0.0])

        # XArm setup
        self.arm = XArmAPI(ip, baud_checkset=False)
        self.arm.clean_warn()
        self.arm.clean_error()
        self.arm.motion_enable(True)
        self.arm.set_mode(0)   # Position control
        self.arm.set_state(0)  # Ready state

        # Gripper
        self.gripper_enable = True
        if self.gripper_enable:
            self.arm.set_gripper_enable(True)
            self.arm.set_gripper_mode(0)
            self.arm.clean_gripper_error()
            # default open
            self.arm.set_gripper_position(850, wait=True)

        # RealSense setup
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(rs_id)
        self.config.enable_stream(rs.stream.color, self.render_width, self.render_height, rs.format.bgr8, 30)
        self.pipeline.start(self.config)
        self.realsense_serial = rs_id


        self.step_counter = 0

        
    def reset(self):        

        self.arm.set_servo_angle(angle=self.init_servo_angle, isradian=False, wait=True)
        if self.gripper_enable:
            self.arm.set_gripper_position(850, wait=True)

        self.step_counter = 0
        obs = self.get_obs()
        return obs, {}

    def step(self, action, mode="ee_pose"):
        assert action.shape == (8,)


        action_angle = radian2angle(action[0:7].tolist())
        self.arm.set_servo_angle(angle=action_angle, isradian=False, wait=False)


        print("action:", action)
        # Gripper control
        if self.gripper_enable:
            # self.arm.set_gripper_position(500, wait=False)
            if action[7] < 0.5:
                self.arm.set_gripper_position(850, wait=False)
            else:
                self.arm.set_gripper_position(np.clip(action[7], 0.5, 2.5), wait=False)

        # Step wait
        self.step_counter += 1

        obs = self.get_obs()
        reward, terminated, truncated = 0.0, False, False
        return obs, reward, terminated, truncated, {}


    def close(self):
        self.pipeline.stop()
        if self.gripper_enable:
            self.arm.set_gripper_position(255, wait=True)  # safe open
        self.arm.disconnect()


    def get_obs(self):
        obs = {}

        # Camera frame
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("No RealSense frame received")
        
        color_image = np.asanyarray(color_frame.get_data())
        H_rs_to_mj = (self.K_mj @ np.linalg.inv(self.K_rs)).astype(np.float32)
        color_bgr = cv2.warpPerspective(color_image, H_rs_to_mj, (640, 480), flags=cv2.INTER_LINEAR)



        obs[f"observation.images.camera_{self.realsense_serial}"] = color_bgr
        obs["observation.state"] = self.get_ee_pose_mj()

        ee_pose_mj = self.get_ee_pose_mj()

        return obs

    def get_ee_pose(self):
        _, qpos = self.arm.get_servo_angle(is_radian=True)
        qpos = np.array(qpos)

        # Forward kinematics → starting EE pose\
        fk_dict = self.kin_helper.compute_fk_from_link_names(np.concatenate([qpos, np.zeros(6)]), ["link_tcp"])
        ee_pose_mat = fk_dict["link_tcp"]
        pos, quat = T.mat2pose(ee_pose_mat, order="xyzw")
        current_pose = np.concatenate([pos, quat])  # shape (7,) x, y, z, qx, qy, qz, qw
        return current_pose