import argparse
import numpy as np
import mujoco
import open3d as o3d
import pyrealsense2 as rs
import os
import pickle

from calibrator import query_mj_cam

def crop_pcd(pcd):
    """
    Crops the point cloud to the specific region of interest:
    x: [-0.2, 0]
    y: [0., 0.2]
    z: [0, 1]
    """
    if pcd is None:
        return None
        
    # Define AABB (Axis Aligned Bounding Box)
    min_bound = np.array([-0.2, 0.0, -1.0])
    max_bound = np.array([ 0.3, 0.2, 0.0])
    
    bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    
    # Crop the point cloud
    cropped_pcd = pcd.crop(bounding_box)
    
    return cropped_pcd

def get_mujoco_pcd(xml_path, camera_name):
    """
    Loads MuJoCo model, resets joints, renders depth, and returns a Blue PointCloud
    transformed to World Coordinates.
    """
    print(f"[MuJoCo] Loading model from: {xml_path}")
    
    # Load Model and Data (No error handling)
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    reset_pose = np.array([0., -0.78539816, 0., 0.52359878, 0., 1.30899694, 0.])
    data.qpos[0:7] = reset_pose

    gripper_target_val = 0.85
    gripper_joint_names = [
        "right_driver_joint", 
        "left_driver_joint",
        "right_finger_joint", 
        "left_finger_joint",
        "right_inner_knuckle_joint", 
        "left_inner_knuckle_joint"
    ]

    for name in gripper_joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        addr = model.jnt_qposadr[jid]
        data.qpos[addr] = gripper_target_val

    mujoco.mj_forward(model, data)

    width, height = 640, 480
    
    # Get Extrinsic (World->Cam) and Intrinsic
    ex_mj, k_mj = query_mj_cam(data, camera_name, width, height)
    
    renderer = mujoco.Renderer(model, height, width)
    renderer.update_scene(data, camera=camera_name)
    renderer.enable_depth_rendering()
    depth_image = renderer.render()
    renderer.disable_depth_rendering()

    fx, fy = k_mj[0, 0], k_mj[1, 1]
    cx, cy = k_mj[0, 2], k_mj[1, 2]
    
    intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
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

    # 3. Crop to ROI
    pcd = crop_pcd(pcd)

    pcd.paint_uniform_color([0, 0, 1]) # RGB for Blue
    
    return pcd

def get_realsense_pcd(calib_path, rs_id):
    """
    Captures depth from RealSense, aligns to RGB, and returns a Green PointCloud
    transformed to World Coordinates using calibration file.
    """
    print(f"[RealSense] Loading calibration from {calib_path}...")

    # Load Calibration (No error handling)
    with open(calib_path, "rb") as f:
        camera_params = pickle.load(f)
        
    params = camera_params[rs_id]
    ex_rs = params["extrinsic"] # World-to-Camera
    k_rs = params["intrinsic"]

    print(f"[RealSense] Connecting to device {rs_id}...")
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_device(rs_id)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)

    # Warmup
    for _ in range(10):
        pipeline.wait_for_frames()

    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    
    # Generate Pointcloud using K_rs
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    o3d_depth = o3d.geometry.Image(depth_image)
    
    width, height = 640, 480
    fx, fy = k_rs[0, 0], k_rs[1, 1]
    cx, cy = k_rs[0, 2], k_rs[1, 2]
    
    intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        o3d_depth,
        intrinsics,
        depth_scale=1000.0, 
        depth_trunc=3.0,
        stride=1
    )
    
    # Transform to World Frame
    cam_to_world = np.linalg.inv(ex_rs)
    pcd.transform(cam_to_world)

    # Crop to ROI
    pcd = crop_pcd(pcd)

    pcd.paint_uniform_color([0, 1, 0]) # RGB for Green

    # Explicit stop (since finally block is removed)
    pipeline.stop()
        
    return pcd

def main():
    parser = argparse.ArgumentParser(description="Fuse MuJoCo and RealSense Pointclouds")
    parser.add_argument("--input_xml_path", type=str, required=True, help="Path to the MuJoCo XML model file")
    parser.add_argument("--input_camera_path", type=str, required=True, help="Path to the calibration pickle file")
    parser.add_argument("--rs_id", type=str, required=True, help="RealSense Camera Serial Number")
    
    args = parser.parse_args()

    camera_name = "camera_" + args.rs_id

    # Get MuJoCo Cloud (Blue)
    mujoco_pcd = get_mujoco_pcd(args.input_xml_path, camera_name)
    
    # Get RealSense Cloud (Green)
    rs_pcd = get_realsense_pcd(args.input_camera_path, args.rs_id)

    # Fuse and Visualize
    geometries_to_draw = []
    
    if mujoco_pcd:
        print(f"[Viz] Added MuJoCo point cloud ({len(mujoco_pcd.points)} points)")
        geometries_to_draw.append(mujoco_pcd)
    else:
        print("[Viz] MuJoCo cloud creation failed.")
    
    if rs_pcd:
        print(f"[Viz] Added RealSense point cloud ({len(rs_pcd.points)} points)")
        geometries_to_draw.append(rs_pcd)
    else:
        print("[Viz] RealSense cloud creation failed.")

    if not geometries_to_draw:
        print("No point clouds generated. Exiting.")
        return

    print("Opening Visualizer... (Close window to exit)")
    
    # Add a coordinate frame for reference (World Origin)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    geometries_to_draw.append(coord_frame)
    
    o3d.visualization.draw_geometries(geometries_to_draw, 
                                      window_name="Fused Pointclouds (Blue: Sim, Green: Real)",
                                      width=1024, height=768)

if __name__ == "__main__":
    main()