import os
import cv2
import time
import yaml
import pickle
import datetime
import numpy as np
from pathlib import Path

import pyrealsense2 as rs
from xarm.wrapper import XArmAPI

from . import transform_utils as T
from .kin_helper import KinHelper
from .utils import get_linear_interpolation_steps, linear_interpolate_poses


def angle2radian(angle):
    arr = np.array(angle, dtype=np.float32)
    return np.deg2rad(arr)

def radian2angle(radian):
    angle = np.array(radian, dtype=np.float32)
    return np.rad2deg(angle)
    
def gripper_norm_2_angle(gripper_norm):
    return 850 - gripper_norm * 1000

def gripper_angle_2_norm(gripper_angle):
    return 0.85 - gripper_angle / 1000

class Xarm7_env():

    def __init__(self, urdf_path, rs_id):
  
        self.init_servo_angle = np.array([0.0, -45.0, 0.0, 30.0, 0.0, 75.0, 0.0])

        self.kin_helper = KinHelper(urdf_path, "link_tcp")

        # XArm setup
        self.arm = XArmAPI("192.168.1.224", baud_checkset=False)
        self.arm.clean_warn()
        self.arm.clean_error()
        self.arm.motion_enable(True)
        self.arm.set_mode(1)   # Position control
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
        # self.pipeline = rs.pipeline()
        # self.config = rs.config()

        # self.config.enable_device(rs_id)
        # self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        # self.pipeline.start(self.config)
        # self.rs_id = rs_id


        self.step_counter = 0

        self.last_gripper_norm_value = 0.
        self.move_threshold = 0.05
        
        
    def reset(self):        
        self.arm.set_mode(0)   # Position control
        self.arm.set_state(0)
        self.arm.set_servo_angle(angle=self.init_servo_angle, isradian=False, wait=True)
        if self.gripper_enable:
            self.arm.set_gripper_position(850, wait=True)
        self.last_qpos = angle2radian(self.init_servo_angle)
        self.last_target_pose = self.get_ee_pose()
        self.step_counter = 0
        obs = self.get_obs()
        self.arm.set_mode(1)   # Position control
        self.arm.set_state(0)
        return obs, {}

    def step(self, action, mode="ee_pose"):
        assert action.shape == (8,)


        # action_angle = radian2angle(action[0:7].tolist())

        fk_dict = self.kin_helper.compute_fk_from_link_names(np.concatenate([action[0:7], np.zeros(6)]), ["link_tcp"])
        ee_pose_mat = fk_dict["link_tcp"]
        pos, quat = T.mat2pose(ee_pose_mat, order="xyzw")
        target_pose = np.concatenate([pos, quat])

        num_steps = 3
        pose_seq = linear_interpolate_poses(self.last_target_pose, target_pose, num_steps)
        
        for pose in pose_seq:

            initial_qpos = np.concatenate([self.last_qpos, np.zeros(6)])
            tf_mat = T.pose2mat((pose[:3], pose[3:]))
            action_radian, _, _ = self.kin_helper.compute_ik_from_mat(initial_qpos, tf_mat)
            action_angle = radian2angle(action_radian[0:7].tolist())
            
            self.arm.set_servo_angle_j(angles=action_angle, isradian=False, wait=False)
            
            self.last_qpos = action_radian[0:7]
            time.sleep(1/90)
        self.last_target_pose = target_pose

        print("\n\n")
        # Gripper control
        if abs(action[7] - self.last_gripper_norm_value) > self.move_threshold:
            print("gripper_norm_2_angle(action[7]):", gripper_norm_2_angle(action[7]))
            self.arm.set_gripper_position(gripper_norm_2_angle(action[7]), wait=False)
            self.last_gripper_norm_value = action[7]
            time.sleep(0.05)

            if self.last_gripper_norm_value > 0.79:
                self.arm.set_gripper_position(-10, wait=False)
                time.sleep(0.05)

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
        # frames = self.pipeline.wait_for_frames()
        # color_frame = frames.get_color_frame()
        # if not color_frame:
        #     raise RuntimeError("No RealSense frame received")
        
        # color_bgr = np.asanyarray(color_frame.get_data())
        # H_rs_to_mj = (self.K_mj @ np.linalg.inv(self.K_rs)).astype(np.float32)
        # color_bgr = cv2.warpPerspective(color_image, H_rs_to_mj, (640, 480), flags=cv2.INTER_LINEAR)

        # obs[f"observation.images.camera_{self.rs_id}"] = color_bgr
        obs["observation.state_ee_pose"] = self.get_ee_pose()
        
        _, qpos = self.arm.get_servo_angle(is_radian=True)
        obs["observation.state_joint_radian"] = qpos
        ee_pose_mj = self.get_ee_pose()

        return obs

    def get_ee_pose(self):
        _, qpos = self.arm.get_servo_angle(is_radian=True)
        qpos = np.array(qpos)

        fk_dict = self.kin_helper.compute_fk_from_link_names(np.concatenate([qpos, np.zeros(6)]), ["link_tcp"])
        ee_pose_mat = fk_dict["link_tcp"]

        
        pos, quat = T.mat2pose(ee_pose_mat, order="xyzw")
        current_pose = np.concatenate([pos, quat])  # shape (7,) x, y, z, qx, qy, qz, qw

        return current_pose