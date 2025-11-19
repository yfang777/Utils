import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd
import av

from ..interface.env import Xarm7_env
from ..interface.kin_helper import KinHelper
from ..interface import transform_utils as T

import threading
from Camera.perception import Perception
from Camera.camera.multi_realsense import MultiRealsense

def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay a df_traj from a parquet file in Xarm7_env."
    )
    parser.add_argument(
        "--parquet_path",
        type=str,
        required=True,
        help="Path to the parquet file containing the trajectory (df_traj).",
    )
    parser.add_argument(
        "--rs_id",
        type=str,
        required=True,
        help="RealSense / camera id or env id passed to Xarm7_env.",
    )
    parser.add_argument(
        "--urdf_path",
        type=str,
        required=True,
        help="Path to the URDF file used to initialize KinHelper.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.02,
        help="Sleep time (seconds) between steps for visualization.",
    )
    parser.add_argument(
        "--root",
        type=str,
    )
    parser.add_argument(
        "--exp_name",
        type=str,
    )
    return parser.parse_args()


# def process_perception_out(perception):
#         # vis = o3d.visualization.Visualizer()
#         # vis.create_window()
#         # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
#         # vis.add_geometry(coordinate)

#         # Set camera view
#         # ctr = vis.get_view_control()
#         # ctr.set_lookat([0, 0, 0])
#         # ctr.set_front([0, -1, 0])
#         # ctr.set_up([0, 0, -1])

#         while perception.alive.value:
#             print(f"[Vis thread] perception out {time.time()}")
#             if not perception.perception_q.empty():
#                 output = perception.perception_q.get()
#             time.sleep(1)


def replay(args):
    df_traj = pd.read_parquet(Path(args.parquet_path))
    actions = np.array(df_traj["action"])
    num_steps = actions.shape[0]

    kin_helper = KinHelper(urdf_path=args.urdf_path, eef_name="link_tcp")

    env = Xarm7_env(urdf_path=args.urdf_path, rs_id = args.rs_id)
    obs, _ = env.reset()
    state_ee_pose = obs["observation.state_ee_pose"]
    qpos = obs["observation.state_joint_radian"]
    print("state_ee_pose:", state_ee_pose)



    # Camera
    realsense = MultiRealsense(
        resolution=(640, 480),
        capture_fps=30,
        enable_color=True,
        enable_depth=True,
        verbose=False
    )

    realsense.start(wait=True)
    realsense.restart_put(start_time=time.time() + 2)
    time.sleep(2)

    perception = Perception(
        root=args.root,
        realsense=realsense, 
        capture_fps=30, # should be the same as the camera fps
        record_fps=30, # 0 for no record
        record_time=55, # in seconds
        exp_name=args.exp_name,
        process_func=None,
        verbose=False
    )

    perception.start()
    perception.set_record_start()
    for i in range(num_steps):

        # actions: ee_pose + gripper_norm [x ,y, z, qx, qy, qz, qw, gripper_norm]
        action_ee_pose = actions[i][:7]

        ee_pose_mat = T.convert_pose_quat2mat(action_ee_pose)
        action_joint_angle, _, _ = kin_helper.compute_ik_from_mat(np.concatenate([qpos, np.zeros(6)]), ee_pose_mat)
        action_gripper_norm = actions[i][7]
        action = np.concatenate([action_joint_angle[:7], [action_gripper_norm]])

        # action to env: joint_angle + gripper_norm
        obs, _, _, _, _  = env.step(action)
        qpos = obs["observation.state_joint_radian"]
        # image = obs[f"observation.images.camera_{args.rs_id}"]


    perception.set_record_stop()
    realsense.stop()
    print("[replay] Finished replaying trajectory.")


def main():
    args = parse_args()
    replay(args)


if __name__ == "__main__":
    main()
