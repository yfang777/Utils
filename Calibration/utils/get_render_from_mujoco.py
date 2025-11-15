import argparse
import os

import numpy as np
import cv2
import mujoco

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str)
    parser.add_argument("--camera_name", type=str)
    parser.add_argument("--img_path", type=str)
    return parser.parse_args()

def main():
    W = 640
    H = 480
    
    args = get_args()
    
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    # set robot
    data.qpos[0: 7] = np.array([ 0., -0.78539816,  0., 0.52359878, 0., 1.30899694, 0.])
    mujoco.mj_forward(model, data)

    with mujoco.Renderer(model, H, W) as renderer:  # (model, width, height)
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera_name)
        renderer.update_scene(data, camera=cam_id)
        img_rgb = renderer.render()  # BGR uint8

    img_bgr = img_rgb[..., ::-1]

    cv2.imwrite(args.img_path, img_bgr)


if __name__ == "__main__":
    main()