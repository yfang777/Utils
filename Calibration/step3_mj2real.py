import numpy as np
import cv2
import argparse


# ----------------------------
# Board geometry
# ----------------------------
def make_board_cells_sim(nx=6, ny=5, square_len=0.038):
    """Return list of (4,3) quads in world coords (meters)."""
    quads = []
    for ix in range(nx):
        for iy in range(ny):
            x0 = ix * square_len
            y0 = -iy * square_len
            x1 = (ix + 1) * square_len
            y1 = -(iy + 1) * square_len
            quad = np.array(
                [
                    [x0, y0, 0.0],
                    [x1, y0, 0.0],
                    [x1, y1, 0.0],
                    [x0, y1, 0.0],
                ],
                dtype=np.float32,
            )
            quads.append(quad)
    return quads

def make_board_cells_real(nx=6, ny=5, square_len=0.038):
    """Return list of (4,3) quads in world coords (meters)."""
    quads = []
    for ix in range(nx):
        for iy in range(ny):
            x0 = ix * square_len
            y0 = -iy * square_len
            x1 = (ix + 1) * square_len
            y1 = -(iy + 1) * square_len
            quad = np.array(
                [
                    [x0, -y0, 0.0],
                    [x1, -y0, 0.0],
                    [x1, -y1, 0.0],
                    [x0, -y1, 0.0],
                ],
                dtype=np.float32,
            )
            quads.append(quad)
    return quads


# ----------------------------
# Camera math
# ----------------------------


def project_points(points, ex, K):
    """
    points: (N,3) world points
    ex: 4x4 extrinsic (world->camera)
    K:  3x3 intrinsic

    Returns: uv (N,2), mask (N,), depth z (N,)
    """

    pts = np.asarray(points, dtype=np.float32)
    N = pts.shape[0]
    pts_h = np.concatenate([pts, np.ones((N, 1), dtype=np.float32)], axis=1)  # (N,4)

    # camera coords
    Xc = (ex @ pts_h.T).T  # (N,3)
    Xc = Xc[:, 0:3]
    z = Xc[:, 2]
    
    # pixel homogeneous
    uvw = (K @ Xc.T).T      # (N,3)
    w = uvw[:, 2]
    valid = (z > 0) & np.isfinite(w) & (np.abs(w) > 1e-12)

    uv = np.full((N, 2), np.nan, dtype=np.float32)
    uv[valid, 0] = uvw[valid, 0] / w[valid]
    uv[valid, 1] = uvw[valid, 1] / w[valid]
    return uv, valid, z


def draw_board(image, quads2d, nx, ny, line_thickness=1):
    """Draw alternating checkerboard squares; skips quads with NaNs."""
    H, W = image.shape[:2]
    for idx, quad in enumerate(quads2d):
        if np.any(~np.isfinite(quad)):
            continue
        ix = idx // ny
        iy = idx % ny
        color = (0, 0, 0) if ((ix + iy) % 2 == 0) else (255, 255, 255)
        poly = np.round(quad).astype(np.int32)
        if np.any((poly[:, 0] < -1000) | (poly[:, 0] > W + 1000) |
                  (poly[:, 1] < -1000) | (poly[:, 1] > H + 1000)):
            continue
        cv2.fillConvexPoly(image, poly, color)
        cv2.polylines(image, [poly], True, (128, 128, 128), line_thickness)


# ----------------------------
# MuJoCo camera helper (world->cam Ex, pixel K)
# ----------------------------
def get_mj_cam(data, camera_name, width, height):
    import mujoco
    import einops as eo

    cam_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    weird_R = eo.rearrange(data.cam_xmat[cam_id], '(i j) -> i j', i=3, j=3).T

    R = np.eye(3, dtype=np.float32)
    R[0, :] = weird_R[0, :]
    R[1, :] = -weird_R[1, :]
    R[2, :] = -weird_R[2, :]
    cam_pos = data.cam_xpos[cam_id].astype(np.float32)
    t = -R @ cam_pos

    Ex = np.eye(4, dtype=np.float32)
    Ex[:3, :3] = R
    Ex[:3, 3]  = t

    fx = data.model.cam_intrinsic[cam_id][0] / data.model.cam_sensorsize[cam_id][0] * data.model.cam_resolution[cam_id][0]
    fy = data.model.cam_intrinsic[cam_id][1] / data.model.cam_sensorsize[cam_id][1] * data.model.cam_resolution[cam_id][1]
    cx = (width  - 1) / 2.0
    cy = (height - 1) / 2.0
    K = np.array([[fx, 0,  cx],
                  [0,  fy, cy],
                  [0,  0,  1 ]], dtype=np.float32)
    return Ex, K


# ----------------------------
# Visualization helpers
# ----------------------------
def render_overlay(H, W, quads3d, Ex, K, base=None, nx=6, ny=5):
    """
    Project board onto base image (or blank) and return the overlay image (H,W,3) BGR.
    """
    if base == None:
        img = np.full((H, W, 3), 220, dtype=np.uint8)
    else:
        img = base

    quads2d = []
    for q in quads3d:
        uv, mask, _ = project_points(q, Ex, K)
        quads2d.append(uv if mask.all() else np.full((4, 2), np.nan, dtype=np.float32))
    draw_board(img, quads2d, nx, ny)
    return img


def save_image(path, img):
    ok = cv2.imwrite(path, img)
    if not ok:
        raise RuntimeError(f"Failed to write image to {path}")


# ----------------------------
# Main
# ----------------------------
def main():
    import pickle
    import mujoco
    import os
    import pyrealsense2 as rs

    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=6)
    parser.add_argument("--ny", type=int, default=5)
    parser.add_argument("--square_len", type=float, default=0.04)
    parser.add_argument("--H", type=int, default=480)
    parser.add_argument("--W", type=int, default=640)
    parser.add_argument("--real_params", type=str, default="/home/yuan/codegen/visual/camera_calibration/calib_results/cam_params.pkl")
    parser.add_argument("--xml", type=str, default="./scene_for_step2.xml")
    parser.add_argument("--real_rgb", type=str, default="", help="Optional path to a real RGB image (BGR is fine).")
    parser.add_argument("--save_prefix", type=str, default="./step3")
    parser.add_argument("--blend_alpha", type=float, default=0.5, help="Alpha for blended overlay (SIM over REAL).")
    args = parser.parse_args()


    # rs_seiral_num = "239222303404"
    # camera_name = "camera_239222303404"

    rs_seiral_num = "239222300412"
    camera_name = "camera_239222300412"

    os.makedirs(args.save_prefix, exist_ok=True)

    # World board
    H, W = args.H, args.W

    # ---- REAL (K, Ex) ----
    with open(args.real_params, "rb") as f:
        params = pickle.load(f)
    print("params:", params)
    Ex_rs = params[rs_seiral_num]["extrinsic"]  # 4x4 world in camera coordinates”
    K_rs  = params[rs_seiral_num]["intrinsic"]  # 3x3

    
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(rs_seiral_num)
 
    config.enable_stream(rs.stream.color, args.W, args.H, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    rgb_rs = np.asanyarray(color_frame.get_data())


    # 1) Visualize real from realsense
    save_image(f"{args.save_prefix}/rgb_rs.png", rgb_rs)

    # 2) Visualize REAL and save
    quads3d = make_board_cells_real(args.nx, args.ny, args.square_len)
    img_real = render_overlay(H, W, quads3d, Ex_rs, K_rs, base=None, nx=args.nx, ny=args.ny)
    save_image(f"{args.save_prefix}/real_calculate.png", img_real)


    blend = cv2.addWeighted(rgb_rs, args.blend_alpha, img_real, 1.0 - args.blend_alpha, 0.0)
    save_image(f"{args.save_prefix}/blend.png", blend)

    print("Done")



if __name__ == "__main__":
    main()
