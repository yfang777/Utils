

import argparse
import os
import time
import pickle
import numpy as np
import cv2
import pyrealsense2 as rs
import mujoco

from calibrator import query_mj_cam

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str)
    parser.add_argument("--img_path", type=str)
    parser.add_argument("--rs_id", type=str)
    parser.add_argument("--rs_params_path", type=str)

    return parser.parse_args()

def main():
    args = get_args()

    W, H = 640, 480
    blend_alpha = 0.5

    # Mujoco
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    Ex_mj, K_mj = query_mj_cam(data, f"camera_{args.rs_id}", W, H)

    # Image
    img_mj_render = cv2.imread(args.img_path, cv2.IMREAD_COLOR)

    # Real
    with open(args.rs_params_path, "rb") as f:
        params = pickle.load(f)
        Ex_rs = params[args.rs_id]["extrinsic"]  # 4x4 world in camera coordinates”
        K_rs  = params[args.rs_id]["intrinsic"]  # 3x3

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(args.rs_id)
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    for _ in range(5):
        pipeline.wait_for_frames()


    win_name = "Real | MuJoCo | Overlay"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            img_rs_bgr = np.asanyarray(color_frame.get_data())
            H_rs_to_mj = (K_mj @ np.linalg.inv(K_rs)).astype(np.float32)
            img_rs_bgr = cv2.warpPerspective(img_rs_bgr, H_rs_to_mj, (W, H), flags=cv2.INTER_LINEAR)

            alpha = float(np.clip(blend_alpha, 0.0, 1.0))
            overlay_bgr = cv2.addWeighted(img_mj_render, alpha, img_rs_bgr, 1.0 - alpha, 0.0)

            left  = img_rs_bgr.copy()
            mid   = img_mj_render.copy()
            right = overlay_bgr.copy()


            strip = np.concatenate([left, mid, right], axis=1)

            cv2.imshow(win_name, strip)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

            if key == ord('s'):
                # Save current RealSense and MuJoCo frames
                cv2.imwrite("assets/figure_save/real.png", img_rs_bgr)
                img__mj_rgb = cv2.cvtColor(img_mj_render, cv2.COLOR_BGR2RGB)
                cv2.imwrite("assets/figure_save/sim.png",  img__mj_rgb)
                print("[saved] real.png and sim.png")

    finally:
        pipeline.stop()
        # renderer.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
