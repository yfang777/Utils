import argparse
import numpy as np
import mujoco
import open3d as o3d
import pyrealsense2 as rs
import os
import pickle
import time
import cv2

from calibrator import query_mj_cam
from utils.env_xarm7 import Xarm7_env, angle2radian

def get_args():
    parser = argparse.ArgumentParser(description="Fuse MuJoCo and RealSense Pointclouds with Robot Control")
    parser.add_argument("--xml", type=str, default="assets/xarm7_gripper/xarm7_with_gripper.xml")
    parser.add_argument("--rs_id", type=str, default="213522071539")
    parser.add_argument("--rs_params_path", type=str, default="assets/calibration_data.pkl")
    parser.add_argument("--rs_frames", type=int, default=200, help="Number of frames to accumulate for RealSense pointcloud")
    return parser.parse_args()

def crop_pcd(pcd):
    """
    Crops the point cloud to the specific region of interest:
    x: [-0.2, 0.3]
    y: [0., 0.2]
    z: [-1.0, 0.0] (Adjusted based on standard offsets, tune as needed)
    """
    if pcd is None:
        return None
        
    # Define AABB (Axis Aligned Bounding Box)
    # Adjust these bounds based on your specific table/workspace setup
    min_bound = np.array([-0.2, -0.4, -1.0])
    max_bound = np.array([ 1.0, 0.5, -0.01])
    
    bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    cropped_pcd = pcd.crop(bounding_box)
    return cropped_pcd

def get_mujoco_pcd(model, data, camera_name, W, H):
    """
    Renders depth from MuJoCo using the current state of 'data' 
    and returns a Blue PointCloud transformed to World Coordinates.
    """
    # 1. Get Extrinsic (World->Cam) and Intrinsic
    ex_mj, k_mj = query_mj_cam(data, camera_name, W, H)
    
    # 2. Render Depth
    renderer = mujoco.Renderer(model, H, W)
    renderer.update_scene(data, camera=camera_name)
    renderer.enable_depth_rendering()
    depth_image = renderer.render()
    renderer.disable_depth_rendering()

    # 3. Create Open3D PointCloud
    fx, fy = k_mj[0, 0], k_mj[1, 1]
    cx, cy = k_mj[0, 2], k_mj[1, 2]
    intrinsics = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
    o3d_depth = o3d.geometry.Image(depth_image)
    
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        o3d_depth, 
        intrinsics, 
        depth_scale=1.0, 
        depth_trunc=10.0, 
        stride=1
    )

    cam_to_world = np.linalg.inv(ex_mj)
    pcd.transform(cam_to_world)

    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    pcd.transform(flip_yz)


    # 5. Crop and Color
    pcd = crop_pcd(pcd)
    pcd.paint_uniform_color([0, 0, 1]) # RGB for Blue
    
    return pcd

def get_realsense_pcd(pipeline, params, W, H, num_frames=30):
    """
    Captures depth from active RealSense pipeline multiple times, 
    aligns to World, merges them, and returns a Green PointCloud.
    """
    ex_rs = params["extrinsic"] # World-to-Camera
    k_rs = params["intrinsic"]

    fx, fy = k_rs[0, 0], k_rs[1, 1]
    cx, cy = k_rs[0, 2], k_rs[1, 2]
    intrinsics = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
    cam_to_world = np.linalg.inv(ex_rs)
    
    combined_pcd = o3d.geometry.PointCloud()
    
    print(f"[RealSense] Accumulating {num_frames} frames...")
    align = rs.align(rs.stream.color)

    for _ in range(num_frames):
        # 1. Capture Frame
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        
        if not aligned_depth_frame:
            continue

        # 2. Create Open3D PointCloud
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        o3d_depth = o3d.geometry.Image(depth_image)
        
        pcd_frame = o3d.geometry.PointCloud.create_from_depth_image(
            o3d_depth,
            intrinsics,
            depth_scale=1000.0,  # RS usually uses mm
            depth_trunc=3.0,
            stride=1
        )
        
        # 3. Transform to World Frame
        pcd_frame.transform(cam_to_world)

        # 4. Crop
        pcd_frame = crop_pcd(pcd_frame)
        
        if pcd_frame is not None:
             combined_pcd += pcd_frame

    # 5. Downsample and Color
    # Voxel downsample to reduce noise and normalize density
    if not combined_pcd.is_empty():
        combined_pcd = combined_pcd.voxel_down_sample(voxel_size=0.005) # 5mm voxel
        combined_pcd.paint_uniform_color([0, 1, 0]) # RGB for Green
        return combined_pcd
    else:
        return None

def main():
    args = get_args()
    W, H = 640, 480

    # ---------------------------
    # 1. Setup MuJoCo
    # ---------------------------
    print(f"[MuJoCo] Loading model from: {args.xml}")
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    camera_name = f"camera_{args.rs_id}"

    # Gripper Joints Setup
    gripper_target_val = 0.85
    gripper_joint_names = [
        "right_driver_joint", "left_driver_joint",
        "right_finger_joint", "left_finger_joint",
        "right_inner_knuckle_joint", "left_inner_knuckle_joint"
    ]

    # ---------------------------
    # 2. Setup RealSense
    # ---------------------------
    print(f"[RealSense] Loading calibration from {args.rs_params_path}")
    with open(args.rs_params_path, "rb") as f:
        camera_params = pickle.load(f)
        
    rs_params = camera_params[args.rs_id]

    print(f"[RealSense] Starting pipeline for device {args.rs_id}...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(args.rs_id)
    config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
    pipeline.start(config)

    # Warmup
    for _ in range(10):
        pipeline.wait_for_frames()

    # ---------------------------
    # 3. Setup Robot Env
    # ---------------------------
    print("[Robot] Connecting to XArm...")
    env_real = Xarm7_env()
    env_real.reset()
    time.sleep(2)

    # ---------------------------
    # 4. Joint Angles List
    # ---------------------------
    joint_angles_list = [
        [-16.6, -24.4, -19.1, 32.3, -9.8, 55.3, -28.8], 
        [12.7, 4.2, 14.1, 61.9, -1.5, 58.4, 27],
    ]

    # ---------------------------
    # 5. Main Loop
    # ---------------------------
    try:
        for i, angles in enumerate(joint_angles_list):
            print(f"\n--- Moving to Pose {i+1} ---")
            print(f"Target Angles (deg): {angles}")

            # A. Move Real Robot
            ee_pose_mat = env_real.compute_ee_pose_from_angles(np.array(angles))
            action = np.concatenate([ee_pose_mat.reshape(-1), [0]]) 
            env_real.step(action=action)
            
            time.sleep(2.0) # Wait for robot to settle

            # B. Move Simulation
            joint_radians = angle2radian(angles)
            data.qpos[0:7] = joint_radians
            # Ensure gripper is set
            for name in gripper_joint_names:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                addr = model.jnt_qposadr[jid]
                data.qpos[addr] = gripper_target_val
            
            mujoco.mj_forward(model, data)

            # C. Capture Point Clouds
            print("[Capture] Generating point clouds...")
            
            mujoco_pcd = get_mujoco_pcd(model, data, camera_name, W, H)
            rs_pcd = get_realsense_pcd(pipeline, rs_params, W, H, num_frames=args.rs_frames)

            # D. Visualize
            geometries = []
            if mujoco_pcd: geometries.append(mujoco_pcd)
            if rs_pcd: geometries.append(rs_pcd)
            
            # Add World Origin
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
            geometries.append(coord_frame)

            print(f"Opening Visualizer for Pose {i+1}. Close window to proceed...")
            o3d.visualization.draw_geometries(geometries, 
                                            window_name=f"Pose {i+1} - Blue: Sim, Green: Real",
                                            width=1024, height=768)

    finally:
        print("Closing resources...")
        pipeline.stop()

if __name__ == "__main__":
    main()