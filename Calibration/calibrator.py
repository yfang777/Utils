import os
import time
import math
import pickle
import numpy as np
import cv2
import pyrealsense2 as rs


class Calibrator:
    def __init__(self):
        # Streams/resolution
        self.W = 640
        self.H = 480

        # Output dirs
        self.vis_dir = "./data/vis"
        self.calibrate_result_dir = "./assets/calib_params"
        os.makedirs(self.vis_dir, exist_ok=True)
        os.makedirs(self.calibrate_result_dir, exist_ok=True)

        # Enumerate all RealSense devices
        ctx = rs.context()
        self.serial_numbers = [dev.get_info(rs.camera_info.serial_number) for dev in ctx.query_devices()]
        self.n_fixed_cameras = len(self.serial_numbers)
        print(f'Found {self.n_fixed_cameras} fixed cameras.')

        # Start one pipeline per device (color+depth)
        self._rs_pipes = {}
        self._rs_profiles = {}
        self._depth_scale = {}
        for serial in self.serial_numbers:
            cfg = rs.config()
            cfg.enable_device(serial)
            
            cfg.enable_stream(rs.stream.color, self.W, self.H, rs.format.bgr8, 30)
            cfg.enable_stream(rs.stream.depth, self.W, self.H, rs.format.z16, 30)

            pipe = rs.pipeline()
            profile = pipe.start(cfg)

            self._rs_pipes[serial] = pipe
            self._rs_profiles[serial] = profile
            depth_sensor = profile.get_device().first_depth_sensor()


            self._depth_scale[serial] = float(depth_sensor.get_depth_scale())

        # ArUco / ChArUco (use *_create for broader OpenCV compatibility)
        self.calibration_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        self.calibration_board = cv2.aruco.CharucoBoard(
            size=(6, 5), squareLength=0.04, markerLength=0.03, dictionary=self.calibration_dictionary,
        )

        self.detector_params = cv2.aruco.DetectorParameters()
        self.charuco_params = cv2.aruco.CharucoParameters()
        self.charuco_detector = cv2.aruco.CharucoDetector(
            self.calibration_board, self.charuco_params, self.detector_params
        )

        # Placeholders for extrinsics (fill later if you estimate world/base)
        self.R_cam2world = None
        self.t_cam2world = None
        self.R_base2world = None
        self.t_base2world = None

    def get_obs(self, get_color=True, get_depth=False) -> dict:
        """
        ONE synchronized frame per camera (depth aligned to color if requested).
        Returns keys: color_{i} [H,W,3 BGR], depth_{i} [H,W float meters], timestamp
        """
        align_to_color = rs.align(rs.stream.color) if get_depth else None
        obs = {}
        timestamp = time.time()

        # Run dummy loop: The first frame usually have high exposure
        for _ in range(1):
            for cam_idx, serial in enumerate(self.serial_numbers):
                pipe = self._rs_pipes[serial]
                frames = pipe.wait_for_frames()
                if get_depth:
                    frames = align_to_color.process(frames)

                if get_color:
                    color_frame = frames.get_color_frame()
                    color_img = np.asanyarray(color_frame.get_data())
                    obs[f"color_{cam_idx}"] = [color_img]

                if get_depth:
                    depth_frame = frames.get_depth_frame()
                    depth_image = np.asanyarray(depth_frame.get_data())
                    depth_scale = self._depth_scale[serial]
                    depth_m = depth_image.astype(np.float32) * depth_scale
                    obs[f"depth_{cam_idx}"] = [depth_m]

        obs["timestamp"] = timestamp
        return obs

    def get_intrinsics(self):
        """
        List of 3x3 K matrices in same order as self.serial_numbers.
        """
        intrinsics = {}
        for serial in self.serial_numbers:
            profile = self._rs_profiles[serial]
            color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            intr = color_stream.get_intrinsics()  # fx, fy, ppx, ppy...
            K = np.array([[intr.fx, 0.0, intr.ppx],
                          [0.0, intr.fy, intr.ppy],
                          [0.0, 0.0, 1.0]], dtype=np.float64)
            intrinsics[serial] = K
        return intrinsics

    def calibrate(self, visualize=True):
        intrinsics = self.get_intrinsics()
        # dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

        board = cv2.aruco.CharucoBoard(
            size=(6, 5), squareLength=0.04, markerLength=0.03, dictionary=dictionary,
        )
        camera_params = {}
        flag = True
        dist_coef = np.zeros(5, dtype=np.float64)

        for i in range(self.n_fixed_cameras):
    

            serial_number = self.serial_numbers[i]
            print("Calibrate on serial_num:", serial_number)
            intrinsic = intrinsics[serial_number]

            c2ws = []

            flag = True
            while flag:
                flag = False

                obs = self.get_obs(get_color=True, get_depth=True)
                
                colors = [obs[f"color_{i}"][0] for i in range(self.n_fixed_cameras)]
                calibration_img = colors[i]
                color_bgr = colors[i]
                
                corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
                    image=calibration_img,
                    dictionary=dictionary,
                    parameters=None,
                )
                retval, charuco_corners, charuco_ids = (
                    cv2.aruco.interpolateCornersCharuco(
                        markerCorners=corners,
                        markerIds=ids,
                        image=calibration_img,
                        board=board,
                        cameraMatrix=intrinsic,
                    )
                )

                rvec = None
                tvec = None
                retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                    charuco_corners,
                    charuco_ids,
                    board,
                    intrinsic,
                    None,
                    rvec=rvec,
                    tvec=tvec,
                )

                # Reproject the points to calculate the error
                reprojected_points, _ = cv2.projectPoints(
                    board.getChessboardCorners()[charuco_ids, :],
                    rvec,
                    tvec,
                    intrinsic,
                    None,
                )

                # Reshape for easier handling
                reprojected_points = reprojected_points.reshape(-1, 2)
                charuco_corners = charuco_corners.reshape(-1, 2)
                
                # Calculate the error
                error = np.sqrt(
                    np.sum((reprojected_points - charuco_corners) ** 2, axis=1)
                ).mean()

                # print("Reprojection Error:", error)

                # error origin 0.2
                if error > 0.35 or len(charuco_corners) < 11:
                    flag = True
                    # print("Please try again.")
            
            print("Success Project")
            R_board2cam = cv2.Rodrigues(rvec)[0]
            t_board2cam = tvec[:, 0]
            w2c = np.eye(4)
            w2c[:3, :3] = R_board2cam
            w2c[:3, 3] = t_board2cam
            c2ws.append(np.linalg.inv(w2c))

            extrinsic = np.eye(4)
            extrinsic[:3, :3] = R_board2cam
            extrinsic[:3, 3] = t_board2cam

            # Save into dict
            camera_params[serial_number] = {
                "extrinsic": extrinsic,
                "intrinsic": intrinsics[serial_number],
            }

            if visualize:
                vis_markers = color_bgr.copy()
                if corners is not None and ids is not None:
                    cv2.aruco.drawDetectedMarkers(vis_markers, corners, ids)
                cv2.imwrite(f'{self.vis_dir}/calibration_detected_marker_{serial_number}.jpg', vis_markers)

                axes_img = color_bgr.copy()
                cv2.drawFrameAxes(axes_img, intrinsic, dist_coef, rvec, tvec, 0.1)
                cv2.imwrite(f"{self.vis_dir}/calibration_result_{serial_number}.jpg", axes_img)


        with open(f'{self.calibrate_result_dir}/cam_params.pkl', "wb") as f:
            print("camera_params:", camera_params)
            pickle.dump(camera_params, f)


def convert_rs_params_to_mj_xml(extrinsic: np.ndarray, intrinsic: np.ndarray, camera_name: str, resolution=(640, 480)) -> str:

    # Decompose extrinsic
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    
    # If your extrinsic is camera_T_world, use this conversion:
    R_c2b = R.T
    t_c2b = -R.T @ t

    R_b2w = np.array([[1, 0, 0],
              [0, -1, 0],
              [0, 0, -1]])
    t_b2w = np.array([0, 0, 0])

    T_c2b = np.eye(4)
    T_c2b[:3, :3] = R_c2b
    T_c2b[:3, 3] = t_c2b

    T_b2w = np.eye(4)
    T_b2w[:3, :3] = R_b2w
    T_b2w[:3, 3] = t_b2w

    # Compose
    T_c2w = T_b2w @ T_c2b

    R_c2w = T_c2w[:3, :3]
    t_c2w = T_c2w[:3, 3]

    # Camera axes in world
    x_axis = (R_c2w[:, 0] / np.linalg.norm(R_c2w[:, 0])).astype(float)
    y_axis = -(R_c2w[:, 1] / np.linalg.norm(R_c2w[:, 1])).astype(float)

    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    w, h = resolution

    w, h = resolution
    sx, sy = 1, 1

    # Convert from pixel units -> MuJoCo units
    fx_mj = fx / (w * sx) 
    fy_mj = fy / (h * sy)

    cx_mj = cx / (w * sx)
    cy_mj = cy / (h * sy)

    xml = (
        f'<camera name="{camera_name}"\n'
        f'  pos="{float(t_c2w[0]):.8f} {float(t_c2w[1]):.8f} {float(t_c2w[2]):.8f}"\n'
        f'  mode="fixed"\n'
        f'  resolution="{w} {h}" sensorsize="{sx} {sy}"\n'
        f'  focal="{fx_mj} {fy_mj}" \n'
        # f'  principalpixel="{cx_mj} {cy_mj}" \n'
        f'  xyaxes="{x_axis[0]:.8f} {x_axis[1]:.8f} {x_axis[2]:.8f}  '
        f'{y_axis[0]:.8f} {y_axis[1]:.8f} {y_axis[2]:.8f}"/>'
    )


    return xml

def add_camera_xml_to_scene_xml(xml_file_path: str, camera_xml: str, output_xml_path: str = None):
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    worldbody = root.find("worldbody")
    new_cam_elem = ET.fromstring(camera_xml)
    worldbody.append(new_cam_elem)

    tree.write(output_xml_path, encoding="utf-8", xml_declaration=True)


def query_mj_cam(data, camera_name, width, height):
    '''
        data: mujoco.model.data
        camera_name: name
        width: 640
        height: 480
    '''
    import mujoco
    from mujoco import Renderer
    import pickle
    import einops as eo


    camera_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    weird_R = eo.rearrange(data.cam_xmat[camera_id], '(i j) -> i j', i=3, j=3).T
    R = np.eye(3)
    R[0, :] = weird_R[0, :]
    R[1, :] = -weird_R[1, :]
    R[2, :] = -weird_R[2, :]
    cam_pos = data.cam_xpos[camera_id]

    t = -np.dot(R, cam_pos)

    ex_mat = np.eye(4)
    ex_mat[:3, :3] = R
    ex_mat[:3, 3] = t

    fx = data.model.cam_intrinsic[camera_id][0] / data.model.cam_sensorsize[camera_id][0] * data.model.cam_resolution[camera_id][0]
    fy = data.model.cam_intrinsic[camera_id][1] / data.model.cam_sensorsize[camera_id][1] * data.model.cam_resolution[camera_id][1]
    cx = (width - 1) / 2
    cy = (height - 1) / 2

    in_mat = np.array([[fx, 0, cx],
                       [0, fy, cy],
                       [0,  0,  1]])

    return ex_mat, in_mat

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rs_id", type=str)
    return parser.parse_args()

def main():

    args = get_args()

    calib = Calibrator()
    calib.calibrate()
    rs_id = args.rs_id
    camera_name = f"camera_{rs_id}"

    with open("./assets/calib_params/cam_params.pkl", "rb") as f:
        camera_params = pickle.load(f)

    params = camera_params[rs_id]
    extrinsic = params["extrinsic"]
    intrinsic = params["intrinsic"]

    camera_xml = convert_rs_params_to_mj_xml(extrinsic, intrinsic, camera_name)
    add_camera_xml_to_scene_xml(xml_file_path="./assets/tmp_scene.xml", camera_xml=camera_xml, output_xml_path="./assets/tmp_scene_w_camera.xml")


if __name__ == "__main__":
    main()
