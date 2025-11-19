import copy
import time
from typing import Optional

import numpy as np

import open3d as o3d
import sapien.core as sapien
from yourdfpy import URDF
from scipy.spatial.transform import Rotation as Rscipy

from . import transform_utils as T


class KinHelper:
    """Helper class for kinematics-related functions for xarm7"""

    def __init__(self, urdf_path, eef_name):
        self.urdf_path = urdf_path # "./assets/xarm7_gripper/xarm7_with_gripper.urdf"
        self.eef_name = eef_name
        self.robot_name = "xarm7_gripper"
        self.active_qmask = np.array([True, True, True, True, True, True, True, False, False, False, False, False, False])


        # load sapien robot
        self.engine = sapien.Engine()
        self.scene = self.engine.create_scene()
        loader = self.scene.create_urdf_loader()
        self.sapien_robot = loader.load(self.urdf_path)
        self.robot_model = self.sapien_robot.create_pinocchio_model()
        self.link_name_to_idx: dict = {}
        for link_idx, link in enumerate(self.sapien_robot.get_links()):
            self.link_name_to_idx[link.name] = link_idx
        self.sapien_eef_idx = self.link_name_to_idx[self.eef_name]
        self.urdf_robot = URDF.load(self.urdf_path)


    def compute_ik_from_mat(
        self,
        initial_qpos: np.ndarray,
        tf_mat: np.ndarray,
        damp: float = 1e-1,
    ) -> tuple[np.ndarray, bool, float]:
        """
        Compute IK given initial joint pos and target pose in matrix form
            Args: 
            * initial_qpos: [13]
            * tf_mat: [3, 3]
            Return:
            * qpos: [13]
            * success: True / False
            * error: [6]
        """

        R = tf_mat[:3, :3]          # rotation matrix
        t = tf_mat[:3, 3]           # translation vector
        quat = Rscipy.from_matrix(R).as_quat()
        quat = T.convert_quat(quat, "wxyz")

        pose = sapien.Pose(p=t, q=quat)

        # pose = sapien.Pose.from_transformation_matrix(tf_mat)

        qpos, success, error = self.robot_model.compute_inverse_kinematics(
            link_index=self.sapien_eef_idx,
            pose=pose,
            initial_qpos=initial_qpos,
            active_qmask=self.active_qmask,
            eps=1e-3,
            damp=damp,
        )
        return qpos, success, error

    def compute_fk_from_link_names(
        self,
        qpos: np.ndarray,
        link_names: list[str],
        in_obj_frame: bool = False,
    ) -> dict[str, np.ndarray]:
        """Compute forward kinematics of robot links given joint positions"""
        self.robot_model.compute_forward_kinematics(qpos)
        link_idx_ls = [self.link_name_to_idx[link_name] for link_name in link_names]
        poses_ls = self.compute_fk_from_link_idx(qpos, link_idx_ls)
        if in_obj_frame:
            for i in range(len(link_names)):
                if link_names[i] in self.offsets:
                    poses_ls[i] = poses_ls[i] @ self.offsets[link_names[i]]
        return {link_name: pose for link_name, pose in zip(link_names, poses_ls)}

    def compute_fk_from_link_idx(
        self,
        qpos: np.ndarray,
        link_idx: list[int],
    ) -> list[np.ndarray]:
        """Compute forward kinematics of robot links given joint positions"""
        self.robot_model.compute_forward_kinematics(qpos)
        link_pose_ls = []
        for i in link_idx:
            pose = self.robot_model.get_link_pose(i)
            link_pose_ls.append(pose.to_transformation_matrix())
        return link_pose_ls