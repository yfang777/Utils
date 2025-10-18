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
        self.vis_dir = "./vis"
        self.calibrate_result_dir = "./calib_results"
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
            if serial_number in ["239222300412"]:
                continue
            print("serial_num:", serial_number)
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

                print("Reprojection Error:", error)

                # error origin 0.2
                if error > 0.35 or len(charuco_corners) < 11:
                    flag = True
                    print("Please try again.")
            
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




    def fixed_camera_calibrate(self, visualize=True, save=True, return_results=True):
        rvecs, tvecs = {}, {}
        rvecs_list, tvecs_list = [], []

        obs = self.get_obs(get_color=True, get_depth=visualize)
        intrinsics = self.get_intrinsics()
        dist_coef = np.zeros(5, dtype=np.float64)



        for i in range(self.n_fixed_cameras):
            serial_number = self.serial_numbers[i]
            print("serial number:", serial_number)
            intrinsic = intrinsics[serial_number]

            color_bgr = obs[f'color_{i}'][-1].copy()
            if visualize:
                cv2.imwrite(f'{self.vis_dir}/calibration_img_{serial_number}.jpg', color_bgr)

            gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)

            charuco_corners, charuco_ids, marker_corners, marker_ids = self.charuco_detector.detectBoard(gray)

            if visualize:
                vis_markers = color_bgr.copy()
                if marker_corners is not None and marker_ids is not None:
                    cv2.aruco.drawDetectedMarkers(vis_markers, marker_corners, marker_ids)
                cv2.imwrite(f'{self.vis_dir}/calibration_detected_marker_{serial_number}.jpg', vis_markers)

                if f'depth_{i}' in obs:
                    depth = obs[f'depth_{i}'][-1].copy()
                    depth = np.minimum(depth, 2.0)
                    maxv = float(depth.max())
                    depth_vis = (depth / maxv * 255.0).astype(np.uint8)
                    depth_vis = cv2.applyColorMap(np.repeat(depth_vis[:, :, None], 3, axis=2), cv2.COLORMAP_JET)
                    cv2.imwrite(f'{self.vis_dir}/calibration_depth_{serial_number}.jpg', depth_vis)

            retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners, 
                charuco_ids, 
                self.calibration_board, 
                cameraMatrix=intrinsic, 
                distCoeffs=dist_coef,
                rvec=None,
                tvec=None,
            )
  
            
            if visualize:
                axes_img = color_bgr.copy()
                cv2.drawFrameAxes(axes_img, intrinsic, dist_coef, rvec, tvec, 0.1)
                cv2.imwrite(f"{self.vis_dir}/calibration_result_{serial_number}.jpg", axes_img)

            rvecs[serial_number] = rvec
            tvecs[serial_number] = tvec
            rvecs_list.append(rvec.reshape(3))
            tvecs_list.append(tvec.reshape(3))

        if save:
            with open(f'{self.calibrate_result_dir}/rvecs.pkl', 'wb') as f:
                pickle.dump(rvecs, f)
            with open(f'{self.calibrate_result_dir}/tvecs.pkl', 'wb') as f:
                pickle.dump(tvecs, f)
            with open(f'{self.calibrate_result_dir}/K.pkl', 'wb') as f:
                pickle.dump(intrinsics, f)

            camera_params = {}
            for cam_id, rvec in rvecs.items():
                tvec = tvecs[cam_id]
                R, _ = cv2.Rodrigues(rvec)

                # Build extrinsic matrix [R | t]
                extrinsic = np.eye(4)
                extrinsic[:3, :3] = R
                extrinsic[:3, 3] = tvec.flatten()

                # Save into dict
                camera_params[cam_id] = {
                    "extrinsic": extrinsic,
                    "intrinsic": intrinsics[cam_id],
                }

            with open(f'{self.calibrate_result_dir}/cam_params.pkl', "wb") as f:
                pickle.dump(camera_params, f)


        if return_results:
            return rvecs, tvecs





def convert_to_mj_camera(extrinsic: np.ndarray, intrinsic: np.ndarray, camera_name, name: str = "camera",
            resolution=(640, 480)) -> str:

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

    print("cx:", cx)
    print("cy:", cy)
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




def replace_camera_in_xml(xml_file_path: str, camera_name: str, camera_xml: str, output_path: str = None):
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("No <worldbody> tag found in XML.")


    new_cam_elem = ET.fromstring(camera_xml)

    replaced = False

    for cam in worldbody.findall("camera"):
        if cam.get("name") == camera_name:
            print(f"Replace Camera {camera_name}")
            print(camera_xml)
            worldbody.remove(cam)
            worldbody.append(new_cam_elem)
            replaced = True
            break

    if not replaced:
        print(f"[WARN] Camera '{camera_name}' not found; adding new camera to <worldbody>.")
        worldbody.append(new_cam_elem)

    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    return ET.tostring(root, encoding="unicode")


def get_mj_cam(data, camera_name, width, height):
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


def test_calibrate(visualize=True, save=True):
    calib = Calibrator()
    calib.calibrate()
    # rvecs, tvecs = calib.fixed_camera_calibrate(
    #     visualize=visualize, save=save, return_results=True
    # )

def test_gen_mj(camera_name):
    # calib = Calibrator()
    # rvecs, tvecs = calib.fixed_camera_calibrate(
    #     visualize=True, save=True, return_results=True
    # )
    
    cam_id = camera_name[7:]
    with open("./calib_results/cam_params.pkl", "rb") as f:
        camera_params = pickle.load(f)

    params = camera_params[cam_id]
    extrinsic = params["extrinsic"]
    intrinsic = params["intrinsic"]

    xml = convert_to_mj_camera(
        extrinsic, intrinsic, camera_name, resolution=(640, 480)
    )
    return xml


def test_replace_camera(camera_name):
    xml = test_gen_mj(camera_name)
    replace_camera_in_xml(
        xml_file_path="./assets/tmp_scene.xml",
        camera_name=camera_name,
        camera_xml=xml,
        output_path="./assets/scene_new.xml"
    )


def compare_sim2real_cube(camera_name):
    import mujoco

    xml_path = "./assets/scene_new.xml"
    cam_id_rs = camera_name[7:]


    # camera_name = "camera_239222300412"
    # cam_id_rs = "239222300412"
    params_path = "./calib_results/cam_params.pkl"

    with open(params_path, "rb") as f:
        camera_params = pickle.load(f)
    K_rs = camera_params[cam_id_rs]["intrinsic"]


    # W = int(K_rs[0, 2] * 2) + 1
    # H = int(K_rs[1, 2] * 2) + 1
    W, H = 640, 480
    W_rs = 640
    H_rs = 480


    print("K_rs:", K_rs)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, H, W) as renderer:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        renderer.update_scene(data, camera=cam_id)
        rgb_mj = renderer.render()
        ex_mat, K_mj = get_mj_cam(data, camera_name, W, H)


    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(cam_id_rs)
 
    config.enable_stream(rs.stream.color, W_rs, H_rs, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    rgb_rs = np.asanyarray(color_frame.get_data())

    
    # Remap the image from realsense intrinsic to mujoco intrinsic
    
    
    # Method 1:
    # map1, map2 = cv2.initUndistortRectifyMap(K_rs, None, None, K_mj, (W, H), cv2.CV_32FC1)
    # c = cv2.remap(rgb_rs, map1, map2, interpolation=cv2.INTER_LINEAR)

    # Method 2:
    # H_rs_to_mj = (K_mj @ np.linalg.inv(K_rs)).astype(np.float32)
    # rgb_rs = cv2.warpPerspective(rgb_rs, H_rs_to_mj, (W, H), flags=cv2.INTER_LINEAR)
    # rgb_mj = cv2.cvtColor(rgb_mj, cv2.COLOR_RGB2BGR)

    # Method 3:
    # color_stream = profile.get_stream(rs.stream.color)
    # intr = color_stream.as_video_stream_profile().get_intrinsics()
    # dist_rs = np.array(intr.coeffs, dtype=np.float64)
    # rgb_rs = cv2.undistort(rgb_rs, K_rs, dist_rs, None, K_rs)

    overlay = cv2.addWeighted(rgb_mj, 0.5, rgb_rs, 1 - 0.5, 0)

    # Save outputs (optional)
    cv2.imwrite("./sim2real_cube/sim_rgb.png", rgb_mj)
    cv2.imwrite("./sim2real_cube/rs_rgb.png", rgb_rs)
    cv2.imwrite("./sim2real_cube/overlay.png", overlay)

    print("Saved sim_rgb.png, rs_rgb_warp.png, overlay.png")

def compare_robot():
    import mujoco
    import transform_utils as T

    

    camera_name = "camera_239222300412"
    cam_id_rs = "239222300412"
    params_path = "./calib_results/cam_params.pkl"
    W = 640
    H = 480

    # Move robot in xarm
    from xarm.wrapper import XArmAPI
    interface="192.168.1.224"
    ArmAPI = XArmAPI(interface, baud_checkset=False)
    init_servo_angle=[0.0, -45.0, 0.0, 30.0, 0.0, 75.0, 0.0]

    ArmAPI.set_servo_angle(angle=init_servo_angle, isradian=False, wait=True)
    time.sleep(2)
    
    ArmAPI.clean_warn()
    ArmAPI._arm.clean_error()
    ArmAPI._arm.motion_enable(True)
    ArmAPI._arm.set_mode(0) # Position control mode
    ArmAPI._arm.set_state(0) # Set to ready state
    ArmAPI.set_gripper_enable(True)
    ArmAPI.set_gripper_mode(0)
    ArmAPI.clean_gripper_error()



    servo_angle = [-12.6, 20.6, -10.8, 79.5, 4.2, 58.6, -25.7]
    # servo_angle = [13.4, -32.8, 8.4, -1.3, 7.2, 31.7, 13.9]
    ArmAPI.set_servo_angle(angle=servo_angle, isradian=False, wait=True)
    time.sleep(2)
    ArmAPI.set_gripper_position(500, wait=True)
    time.sleep(2)
    
    # Move robot in sim
    def angle2radian(angle):
        """
        Convert degrees to radians.
        angle: float, list, np.array (1D/2D)
        """
        arr = np.array(angle, dtype=np.float32)
        return np.deg2rad(arr)
    xml_path = "./assets/scene_new_robot.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    qpos = angle2radian(servo_angle)
    data.qpos[:len(qpos)] = qpos
    mujoco.mj_forward(model, data)
    
    with mujoco.Renderer(model, H, W) as renderer:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        renderer.update_scene(data, camera=cam_id)
        rgb_mj = renderer.render()
        ex_mat, K_mj = get_mj_cam(data, camera_name, W, H)


    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(cam_id_rs)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)


    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    rgb_rs_1 = np.asanyarray(color_frame.get_data())


    with open(params_path, "rb") as f:
        camera_params = pickle.load(f)
    K_rs = camera_params[cam_id_rs]["intrinsic"]
    

    H_rs_to_mj = (K_mj @ np.linalg.inv(K_rs)).astype(np.float32)
    rgb_rs_1 = cv2.warpPerspective(rgb_rs_1, H_rs_to_mj, (W, H), flags=cv2.INTER_LINEAR)

    

    ArmAPI.set_servo_angle(angle=init_servo_angle, isradian=False, wait=True)
    time.sleep(2)
    ArmAPI.set_servo_angle(angle=servo_angle, isradian=False, wait=True)
    time.sleep(2)
    ArmAPI.set_gripper_position(700, wait=True)
    time.sleep(2)

    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    rgb_rs_2 = np.asanyarray(color_frame.get_data())
    rgb_rs_2 = cv2.warpPerspective(rgb_rs_2, H_rs_to_mj, (W, H), flags=cv2.INTER_LINEAR)
    overlay = cv2.addWeighted(rgb_rs_1, 0.5, rgb_rs_2, 1 - 0.5, 0)


    # Save outputs (optional)
    cv2.imwrite("./robot_calibration/rs_rgb_1.png", rgb_rs_1)
    cv2.imwrite("./robot_calibration/rs_rgb_2.png", rgb_rs_2)
    cv2.imwrite("./robot_calibration/overlay.png", overlay)

    print("Saved sim_rgb.png, rs_rgb_warp.png, overlay.png")


def compare_sim2real_robot(camera_name):
    import mujoco
    import transform_utils as T
    from ik_solver_sapien import KinHelper

    xml_path = "./assets/scene_new_robot.xml"
    cam_id_rs = camera_name[7:]
    params_path = "./calib_results/cam_params.pkl"
    W = 640
    H = 480

    # Move robot in xarm
    from xarm.wrapper import XArmAPI
    interface="192.168.1.224"
    ArmAPI = XArmAPI(interface, baud_checkset=False)
    init_servo_angle=[0.0, -45.0, 0.0, 30.0, 0.0, 75.0, 0.0]

    kin_helper = KinHelper("link_tcp")
    ArmAPI.set_servo_angle(angle=init_servo_angle, isradian=False, wait=True)
    time.sleep(2)

    ArmAPI.clean_warn()
    ArmAPI._arm.clean_error()
    ArmAPI._arm.motion_enable(True)
    ArmAPI._arm.set_mode(0) # Position control mode
    ArmAPI._arm.set_state(0) # Set to ready state

    # servo_angle = [-10.2, -24.3, -11.3, 29.6, -6.3, 53.2, -17.1]
    # servo_angle = [-11.8, 5.6, -14.9, 59.3, 1.4, 53.2, -28.1]
    # servo_angle = [-12, 14.9, -12.5, 71.4, 3.5, 56.2, -26.8]
    servo_angle = [-12.6, 20.6, -10.8, 79.5, 4.2, 58.6, -25.7]

    # servo_angle = [13.4, -32.8, 8.4, -1.3, 7.2, 31.7, 13.9]
    ArmAPI.set_servo_angle(angle=servo_angle, isradian=False, wait=True)
    
    # Move robot in sim
    def angle2radian(angle):
        """
        Convert degrees to radians.
        angle: float, list, np.array (1D/2D)
        """
        arr = np.array(angle, dtype=np.float32)
        return np.deg2rad(arr)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    qpos = angle2radian(servo_angle)
    data.qpos[:len(qpos)] = qpos

    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, H, W) as renderer:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        renderer.update_scene(data, camera=cam_id)
        rgb_mj = renderer.render()
        ex_mat, K_mj = get_mj_cam(data, camera_name, W, H)


    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(cam_id_rs)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    rgb_rs = np.asanyarray(color_frame.get_data())


    with open(params_path, "rb") as f:
        camera_params = pickle.load(f)
    K_rs = camera_params[cam_id_rs]["intrinsic"]
    
    # Method 1:
    # map1, map2 = cv2.initUndistortRectifyMap(K_rs, None, None, K_mj, (W, H), cv2.CV_32FC1)
    # c = cv2.remap(rgb_rs, map1, map2, interpolation=cv2.INTER_LINEAR)
    # img_rect = cv2.undistort(rgb_rs, K_rs, dist_rs, None, K_rs)


    # Method 2:
    H_rs_to_mj = (K_mj @ np.linalg.inv(K_rs)).astype(np.float32)
    rgb_rs = cv2.warpPerspective(rgb_rs, H_rs_to_mj, (W, H), flags=cv2.INTER_LINEAR)
    overlay = cv2.addWeighted(rgb_mj, 0.5, rgb_rs, 1 - 0.5, 0)


    # Save outputs (optional)
    cv2.imwrite("./sim2real_robot/sim_rgb.png", cv2.cvtColor(rgb_mj, cv2.COLOR_RGB2BGR))
    cv2.imwrite("./sim2real_robot/rs_rgb.png", rgb_rs)
    cv2.imwrite("./sim2real_robot/overlay.png", overlay)

    print("Saved sim_rgb.png, rs_rgb_warp.png, overlay.png")


def main():
    
    # test_calibrate(visualize=True, save=True)
    # test_gen_mj()
    camera_name = "camera_235422302222"
    # test_replace_camera(camera_name)
    # compare_sim2real_cube(camera_name)
    compare_sim2real_robot(camera_name)
    # compare_robot()


if __name__ == "__main__":
    main()
