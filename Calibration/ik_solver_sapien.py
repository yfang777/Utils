import copy
import time
from typing import Optional

import numpy as np

import open3d as o3d
import sapien.core as sapien
import transforms3d
import trimesh
from yourdfpy import URDF
from scipy.spatial.transform import Rotation as Rscipy

import transform_utils as T


def trimesh_to_open3d(tri_mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(tri_mesh.vertices)
    mesh.triangles = o3d.utility.Vector3iVector(tri_mesh.faces)
    if tri_mesh.visual.kind == 'vertex':
        mesh.vertex_colors = o3d.utility.Vector3dVector(tri_mesh.visual.vertex_colors[:, :3] / 255.0)
    elif tri_mesh.visual.kind == 'face':
        mesh.triangle_colors = o3d.utility.Vector3dVector(tri_mesh.visual.face_colors[:, :3] / 255.0)
    return mesh

class KinHelper:
    """Helper class for kinematics-related functions for xarm7"""

    def __init__(self, eef_name):
        self.urdf_path = "./assets/xarm7_gripper/xarm7_with_gripper.urdf"
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

        # load meshes and offsets from urdf_robot
        self.urdf_robot = URDF.load(self.urdf_path)
        self.meshes = {}
        self.scales = {}
        self.offsets = {}
        for link_name, link in self.urdf_robot.link_map.items():
            if len(link.collisions) > 0:
                collision = link.collisions[0]
                if (
                    collision.geometry.mesh is not None
                    # and len(collision.geometry.mesh.meshes) > 0
                ):
                    mesh_path = self.urdf_robot._filename_handler(collision.geometry.mesh.filename)
                    mesh_scale = collision.geometry.mesh.scale
                    mesh = trimesh.load(mesh_path)
                    self.meshes[link.name] = trimesh_to_open3d(mesh)
                    self.meshes[link.name].compute_vertex_normals()
                    self.meshes[link.name].paint_uniform_color([0.2, 0.2, 0.2])
                    self.scales[link.name] = (
                        mesh_scale
                        if collision.geometry.mesh.scale is not None
                        else 1.0
                    )
                    self.offsets[link.name] = collision.origin
        self.pcd_dict: dict = {}
        self.tool_meshes: dict = {}

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

    def compute_all_fk(
        self, qpos: np.ndarray, in_obj_frame: bool = False
    ) -> dict[str, np.ndarray]:
        """Compute forward kinematics of all robot links given joint positions"""
        all_link_names = [link.name for link in self.sapien_robot.get_links()]
        print("all_link_names:", all_link_names)
        return self.compute_fk_from_link_names(qpos, all_link_names, in_obj_frame)

    def compute_ik_euler(
        self,
        initial_qpos: np.ndarray,
        cartesian: np.ndarray,
        damp: float = 1e-1,
    ) -> np.ndarray:
        """Compute inverse kinematics given initial joint pos and target pose"""
        tf_mat = np.eye(4)
        tf_mat[:3, :3] = transforms3d.euler.euler2mat(
            ai=cartesian[3], aj=cartesian[4], ak=cartesian[5], axes="sxyz"
        )
        tf_mat[:3, 3] = cartesian[0:3]
        pose = sapien.Pose.from_transformation_matrix(tf_mat)
        
        qpos, success, error = self.robot_model.compute_inverse_kinematics(
            link_index=self.sapien_eef_idx,
            pose=pose,
            initial_qpos=initial_qpos,
            active_qmask=self.active_qmask,
            eps=1e-3,
            damp=damp,
        )
        return qpos, success, error

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


    def _mesh_poses_to_pc(
        self,
        poses: np.ndarray,
        meshes: list[o3d.geometry.TriangleMesh],
        offsets: list[np.ndarray],
        num_pts: list[int],
        scales: list[int],
        pcd_name: Optional[str] = None,
    ) -> np.ndarray:
        # poses: (N, 4, 4) numpy array
        # offsets: (N, ) list of offsets
        # meshes: (N, ) list of meshes
        # num_pts: (N, ) list of int
        # scales: (N, ) list of float
        try:
            assert poses.shape[0] == len(meshes)
            assert poses.shape[0] == len(offsets)
            assert poses.shape[0] == len(num_pts)
            assert poses.shape[0] == len(scales)
        except AssertionError:
            print("Input shapes do not match")
            exit(1)

        N = poses.shape[0]
        all_pc = []
        for index in range(N):
            mat = poses[index]
            if (
                pcd_name is None
                or pcd_name not in self.pcd_dict
                or len(self.pcd_dict[pcd_name]) <= index
            ):
                mesh = copy.deepcopy(meshes[index])  # .copy()
                mesh.scale(scales[index], center=np.array([0, 0, 0]))
                sampled_cloud = mesh.sample_points_poisson_disk(
                    number_of_points=num_pts[index]
                )
                cloud_points = np.asarray(sampled_cloud.points)
                if pcd_name not in self.pcd_dict:
                    self.pcd_dict[pcd_name] = []
                self.pcd_dict[pcd_name].append(cloud_points)
            else:
                cloud_points = self.pcd_dict[pcd_name][index]

            tf_obj_to_link = offsets[index]

            mat = mat @ tf_obj_to_link
            transformed_points = cloud_points @ mat[:3, :3].T + mat[:3, 3]
            all_pc.append(transformed_points)
        all_pc = np.concatenate(all_pc, axis=0)
        return all_pc

    def compute_robot_pcd(
        self,
        qpos: np.ndarray,
        link_names: Optional[list[str]] = None,
        num_pts: Optional[list[int]] = None,
        pcd_name: Optional[str] = None,
    ) -> np.ndarray:
        """Compute point cloud of robot links given joint positions"""
        self.robot_model.compute_forward_kinematics(qpos)
        if link_names is None:
            link_names = list(self.meshes.keys())

        if num_pts is None:
            num_pts = [500] * len(link_names)
        link_idx_ls = []
        for link_name in link_names:
            for link_idx, link in enumerate(self.sapien_robot.get_links()):
                if link.name == link_name:
                    link_idx_ls.append(link_idx)
                    break

        link_pose_ls = np.stack(
            [
                self.robot_model.get_link_pose(link_idx).to_transformation_matrix()
                for link_idx in link_idx_ls
            ]
        )
        meshes_ls = [self.meshes[link_name] for link_name in link_names]
        offsets_ls = [self.offsets[link_name] for link_name in link_names]
        scales_ls = [self.scales[link_name] for link_name in link_names]
        pcd = self._mesh_poses_to_pc(
            poses=link_pose_ls,
            meshes=meshes_ls,
            offsets=offsets_ls,
            num_pts=num_pts,
            scales=scales_ls,
            pcd_name=pcd_name,
        )
        return pcd

    def compute_robot_meshes(
        self,
        qpos: np.ndarray,
        link_names: Optional[list[str]] = None,
    ) -> list[o3d.geometry.TriangleMesh]:
        """Compute meshes of robot links given joint positions"""
        self.robot_model.compute_forward_kinematics(qpos)
        if link_names is None:
            link_names = list(self.meshes.keys())
        link_idx_ls = []
        for link_name in link_names:
            for link_idx, link in enumerate(self.sapien_robot.get_links()):
                if link.name == link_name:
                    link_idx_ls.append(link_idx)
                    break
        link_pose_ls = np.stack(
            [
                self.robot_model.get_link_pose(link_idx).to_transformation_matrix()
                for link_idx in link_idx_ls
            ]
        )
        offsets_ls = [self.offsets[link_name] for link_name in link_names]
        meshes_ls = []
        for link_idx, link_name in enumerate(link_names):
            import copy

            mesh = copy.deepcopy(self.meshes[link_name])
            mesh.scale(0.001, center=np.array([0, 0, 0]))
            tf = link_pose_ls[link_idx] @ offsets_ls[link_idx]
            mesh.transform(tf)
            meshes_ls.append(mesh)
        return meshes_ls

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

    



