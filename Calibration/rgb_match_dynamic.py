import argparse
import os
import time
import pickle
import numpy as np
import cv2
import pyrealsense2 as rs
import mujoco

from calibrator import query_mj_cam
from utils.env_xarm7 import Xarm7_env, angle2radian
from utils.ik_solver_sapien import KinHelper

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, default="assets/xarm7_gripper/xarm7_with_gripper.xml")
    parser.add_argument("--rs_id", type=str, default="213522071539")
    parser.add_argument("--rs_params_path", type=str, default="assets/calibration_data.pkl")

    return parser.parse_args()

def capture_image(image_name, pipeline, model, data, camera_name, K_mj, K_rs, W, H):
    """
    Captures RealSense frame, Renders MuJoCo frame, overlaps them, and saves all three.
    """
    # 1. Capture RealSense Frame
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        print("Warning: No RealSense frame received.")
        return

    img_rs_bgr = np.asanyarray(color_frame.get_data())

    # 2. Render MuJoCo Frame
    with mujoco.Renderer(model, H, W) as renderer:
        renderer.update_scene(data, camera=camera_name)
        img_mj_render = renderer.render() # Returns RGB usually, check format
        img_mj_bgr = cv2.cvtColor(img_mj_render, cv2.COLOR_RGB2BGR)

    # 3. Warp RealSense to match MuJoCo Intrinsics (Simple Homography)
    # Note: This simple homography assumes cameras are rotationally aligned or only differ by intrinsics scaling.
    # For full 3D alignment, extrinsic reprojection is required.
    H_rs_to_mj = (K_mj @ np.linalg.inv(K_rs)).astype(np.float32)
    img_rs_warped = cv2.warpPerspective(img_rs_bgr, H_rs_to_mj, (W, H), flags=cv2.INTER_LINEAR)

    # 4. Create Overlay
    alpha = 0.5
    overlay_bgr = cv2.addWeighted(img_mj_bgr, alpha, img_rs_warped, 1.0 - alpha, 0.0)

    # 5. Save Images
    save_dir = "data/figure_save"
    os.makedirs(save_dir, exist_ok=True)
    
    path_real = os.path.join(save_dir, f"{image_name}_real.png")
    path_sim = os.path.join(save_dir, f"{image_name}_sim.png")
    path_overlay = os.path.join(save_dir, f"{image_name}_overlay.png")

    cv2.imwrite(path_real, img_rs_warped)
    cv2.imwrite(path_sim, img_mj_bgr)
    cv2.imwrite(path_overlay, overlay_bgr)

    print(f"[Saved] {image_name} -> Real, Sim, Overlay")

def main():
    args = get_args()

    W, H = 640, 480
    
    # ---------------------------
    # 1. Setup MuJoCo
    # ---------------------------
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    
    gripper_target_val = 0.85
    gripper_joint_names = [
        "right_driver_joint", "left_driver_joint",
        "right_finger_joint", "left_finger_joint",
        "right_inner_knuckle_joint", "left_inner_knuckle_joint"
    ]
    for name in gripper_joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        addr = model.jnt_qposadr[jid]
        data.qpos[addr] = gripper_target_val
    
    mujoco.mj_forward(model, data)
    
    # Get Camera Info
    camera_name = f"camera_{args.rs_id}"
    Ex_mj, K_mj = query_mj_cam(data, camera_name, W, H)

    # ---------------------------
    # 2. Setup RealSense
    # ---------------------------
    with open(args.rs_params_path, "rb") as f:
        params = pickle.load(f)
        Ex_rs = params[args.rs_id]["extrinsic"]
        K_rs  = params[args.rs_id]["intrinsic"]

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(args.rs_id)
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
    pipeline.start(config)

    # Warmup
    for _ in range(10):
        pipeline.wait_for_frames()

    # ---------------------------
    # 3. Setup Robot Env
    # ---------------------------
    env_real = Xarm7_env()
    obs, _ = env_real.reset()
    time.sleep(2)

    # ---------------------------
    # 4. Joint Angles Extraction
    # ---------------------------
    joint_angles_list = [
        [-16.6, -24.4, -19.1, 32.3, -9.8, 55.3, -28.8], 
        [12.7, 4.2, 14.1, 61.9, -1.5, 58.4, 27],
    ]

    image_names = ["pose_1", "pose_2"]

    # ---------------------------
    # 5. Main Loop
    # ---------------------------
    for i, angles in enumerate(joint_angles_list):
        print(f"\n--- Moving to Pose {i+1} ---")
        print(f"Target Angles (deg): {angles}")

        ee_pose_mat = env_real.compute_ee_pose_from_angles(np.array(angles))

        action = np.concatenate([ee_pose_mat.reshape(-1), [0]]) 
        env_real.step(action=action)
        
        # Wait for robot to settle
        time.sleep(2.0)

        # B. Move Simulation
        joint_radians = angle2radian(angles)
        data.qpos[0:7] = joint_radians
        for name in gripper_joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            addr = model.jnt_qposadr[jid]
            data.qpos[addr] = gripper_target_val

        mujoco.mj_forward(model, data)

        # C. Capture and Save
        capture_image(
            image_name=image_names[i],
            pipeline=pipeline,
            model=model,
            data=data,
            camera_name=camera_name,
            K_mj=K_mj,
            K_rs=K_rs,
            W=W,
            H=H
        )

    # Cleanup
    pipeline.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()