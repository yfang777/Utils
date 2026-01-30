
import os
import time
import math
import pickle
import numpy as np
import cv2
import pyzed.sl as sl
import xml.etree.ElementTree as ET

class Calibrator:
    def __init__(self):
        # Streams/resolution - ZED uses specific resolutions
        # HD720 is 1280x720, VGA is 672x376. 
        # We will use HD720 for better quality calibration, or VGA if performance is key.
        # Let's stick to a standard resolution. 
        # The original code used 640x480. VGA (672x376) is closest but aspect ratio differs.
        # HD720 is safe.
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = sl.RESOLUTION.HD720
        self.init_params.camera_fps = 30
        self.init_params.depth_mode = sl.DEPTH_MODE.NONE # We mostly need RGB for calibration, depth optional? 
        # original code got depth if requested. 'get_obs' has 'get_depth' arg.
        # Let's enable depth just in case, but low quality for speed if possible.
        self.init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE 
        self.init_params.coordinate_units = sl.UNIT.METER

        self.W = 1280
        self.H = 720

        # Output dirs
        self.vis_dir = "./data/vis"
        self.calibrate_result_dir = "./assets/calib_params"
        os.makedirs(self.vis_dir, exist_ok=True)
        os.makedirs(self.calibrate_result_dir, exist_ok=True)

        # Enumerate all ZED devices
        self.cams = []
        self.serial_numbers = []
        
        dev_list = sl.Camera.get_device_list()
        self.n_fixed_cameras = len(dev_list)
        print(f'Found {self.n_fixed_cameras} fixed cameras.')

        for dev in dev_list:
            cam = sl.Camera()
            input_type = sl.InputType()
            input_type.set_from_serial_number(dev.serial_number)
            self.init_params.input = input_type
            
            status = cam.open(self.init_params)
            if status != sl.ERROR_CODE.SUCCESS:
                print(f"Failed to open camera {dev.serial_number}: {status}")
                continue
            
            self.cams.append(cam)
            self.serial_numbers.append(str(dev.serial_number))
            
        print(f"Successfully opened {len(self.cams)} cameras: {self.serial_numbers}")

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
        ONE synchronized frame per camera.
        Returns keys: color_{i} [H,W,3 BGR], depth_{i} [H,W float meters], timestamp
        """
        obs = {}
        timestamp = time.time()
        
        # Runtime objects
        runtime_params = sl.RuntimeParameters()
        
        mat_image = sl.Mat()
        mat_depth = sl.Mat()

        # Iterate over open cameras
        for cam_idx, cam in enumerate(self.cams):
            if cam.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                
                if get_color:
                    cam.retrieve_image(mat_image, sl.VIEW.LEFT)
                    # get_data() returns BGRA, we usually want BGR or RGB. 
                    # RealSense code returned BGR.
                    # ZED Python API get_data returns numpy array.
                    img_bgra = mat_image.get_data()
                    img_bgr = img_bgra[:, :, :3]
                    # Make a copy to ensure it's contiguous/safe if needed, or just use slice
                    obs[f"color_{cam_idx}"] = [np.ascontiguousarray(img_bgr)]

                if get_depth:
                    cam.retrieve_measure(mat_depth, sl.MEASURE.DEPTH)
                    depth_map = mat_depth.get_data()
                    # ZED depth is already in meters (if configured so) and float32
                    obs[f"depth_{cam_idx}"] = [np.ascontiguousarray(depth_map)]
            else:
                print(f"Warning: Failed to grab frame from camera {cam_idx}")

        obs["timestamp"] = timestamp
        return obs

    def get_intrinsics(self):
        """
        List of 3x3 K matrices in same order as self.serial_numbers.
        """
        intrinsics = {}
        for idx, cam in enumerate(self.cams):
            serial = self.serial_numbers[idx]
            cam_info = cam.get_camera_information()
            # Left camera intrinsics
            # fx, fy, cx, cy are available directly or via calibration_parameters
            params = cam_info.camera_configuration.calibration_parameters.left_cam
            fx = params.fx
            fy = params.fy
            cx = params.cx
            cy = params.cy
            
            K = np.array([[fx, 0.0, cx],
                          [0.0, fy, cy],
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


ROBOT_XML = os.path.abspath("./assets/xarm7_gripper/xarm7_with_gripper.xml")

TEST_XML = f'''
<mujoco model="scene">
    <include file="{ROBOT_XML}"/>
    <worldbody>
            <light diffuse="0.8 0.8 0.8" 
                specular="0.2 0.2 0.2" 
                pos="0 0 4" 
                dir="0 0 -1" 
                cutoff="180"
                castshadow="false"
                directional="true"/>

            <!-- Gray ground plane -->
            <geom type="plane" 
                size="5 5 0.1" 
                rgba="0.3 0.3 0.3 1" 
                contype="1" 
                conaffinity="1" 
                friction="1 0.005 0.0001"/>
            
            <body name="board_sites" pos="0 0 0">
                <site name="s_00" type="sphere" size="0.003" pos="0.00  0.00  0.0" rgba="1 0 0 1"/>
                <site name="s_10" type="sphere" size="0.003" pos="0.04  0.00  0.0" rgba="1 0 0 1"/>
                <site name="s_20" type="sphere" size="0.003" pos="0.08  0.00  0.0" rgba="1 0 0 1"/>
                <site name="s_30" type="sphere" size="0.003" pos="0.12  0.00  0.0" rgba="1 0 0 1"/>
                <site name="s_40" type="sphere" size="0.003" pos="0.16  0.00  0.0" rgba="1 0 0 1"/>
                <site name="s_50" type="sphere" size="0.003" pos="0.20  0.00  0.0" rgba="1 0 0 1"/>
                <site name="s_60" type="sphere" size="0.003" pos="0.24  0.00  0.0" rgba="1 0 0 1"/>

                <site name="s_01" type="sphere" size="0.003" pos="0.00 -0.04 0.0" rgba="0 1 0 1"/>
                <site name="s_11" type="sphere" size="0.003" pos="0.04 -0.04 0.0" rgba="0 1 0 1"/>
                <site name="s_21" type="sphere" size="0.003" pos="0.08 -0.04 0.0" rgba="0 1 0 1"/>
                <site name="s_31" type="sphere" size="0.003" pos="0.12 -0.04 0.0" rgba="0 1 0 1"/>
                <site name="s_41" type="sphere" size="0.003" pos="0.16 -0.04 0.0" rgba="0 1 0 1"/>
                <site name="s_51" type="sphere" size="0.003" pos="0.20 -0.04 0.0" rgba="0 1 0 1"/>
                <site name="s_61" type="sphere" size="0.003" pos="0.24 -0.04 0.0" rgba="0 1 0 1"/>

                <site name="s_02" type="sphere" size="0.003" pos="0.00 -0.08 0.0" rgba="0 0 1 1"/>
                <site name="s_12" type="sphere" size="0.003" pos="0.04 -0.08 0.0" rgba="0 0 1 1"/>
                <site name="s_22" type="sphere" size="0.003" pos="0.08 -0.08 0.0" rgba="0 0 1 1"/>
                <site name="s_32" type="sphere" size="0.003" pos="0.12 -0.08 0.0" rgba="0 0 1 1"/>
                <site name="s_42" type="sphere" size="0.003" pos="0.16 -0.08 0.0" rgba="0 0 1 1"/>
                <site name="s_52" type="sphere" size="0.003" pos="0.20 -0.08 0.0" rgba="0 0 1 1"/>
                <site name="s_62" type="sphere" size="0.003" pos="0.24 -0.08 0.0" rgba="0 0 1 1"/>

                <site name="s_03" type="sphere" size="0.003" pos="0.00 -0.12 0.0" rgba="1 0 1 1"/>
                <site name="s_13" type="sphere" size="0.003" pos="0.04 -0.12 0.0" rgba="1 0 1 1"/>
                <site name="s_23" type="sphere" size="0.003" pos="0.08 -0.12 0.0" rgba="1 0 1 1"/>
                <site name="s_33" type="sphere" size="0.003" pos="0.12 -0.12 0.0" rgba="1 0 1 1"/>
                <site name="s_43" type="sphere" size="0.003" pos="0.16 -0.12 0.0" rgba="1 0 1 1"/>
                <site name="s_53" type="sphere" size="0.003" pos="0.20 -0.12 0.0" rgba="1 0 1 1"/>
                <site name="s_63" type="sphere" size="0.003" pos="0.24 -0.12 0.0" rgba="1 0 1 1"/>

                <site name="s_04" type="sphere" size="0.003" pos="0.00 -0.16 0.0" rgba="0 1 1 1"/>
                <site name="s_14" type="sphere" size="0.003" pos="0.04 -0.16 0.0" rgba="0 1 1 1"/>
                <site name="s_24" type="sphere" size="0.003" pos="0.08 -0.16 0.0" rgba="0 1 1 1"/>
                <site name="s_34" type="sphere" size="0.003" pos="0.12 -0.16 0.0" rgba="0 1 1 1"/>
                <site name="s_44" type="sphere" size="0.003" pos="0.16 -0.16 0.0" rgba="0 1 1 1"/>
                <site name="s_54" type="sphere" size="0.003" pos="0.20 -0.16 0.0" rgba="0 1 1 1"/>
                <site name="s_64" type="sphere" size="0.003" pos="0.24 -0.16 0.0" rgba="0 1 1 1"/>

            </body>

        </worldbody>
</mujoco>
'''        

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

def add_camera_xml_to_scene_xml(input_xml: str, camera_xml: str, output_xml_path: str):
    """
    Parses xml_content string, appends the camera_xml, and writes to output_xml_path.
    """
    root = ET.fromstring(input_xml)

    worldbody = root.find("worldbody")
    new_cam_elem = ET.fromstring(camera_xml)

    worldbody.append(new_cam_elem)

    tree = ET.ElementTree(root)
    
    output_dir = os.path.dirname(output_xml_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    tree.write(output_xml_path, encoding="utf-8", xml_declaration=True)
    print(f"Saved scene to: {os.path.abspath(output_xml_path)}")



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
    parser.add_argument("--zed_serial", type=str, help="Serial number of the ZED camera")
    return parser.parse_args()

def main():

    args = get_args()

    calib = Calibrator()
    calib.calibrate()
    zed_serial = args.zed_serial
    camera_name = f"camera_{zed_serial}"

    with open("./assets/calib_params/cam_params.pkl", "rb") as f:
        camera_params = pickle.load(f)

    params = camera_params[zed_serial]
    extrinsic = params["extrinsic"]
    intrinsic = params["intrinsic"]

    camera_xml = convert_rs_params_to_mj_xml(extrinsic, intrinsic, camera_name, resolution=(calib.W, calib.H))
    add_camera_xml_to_scene_xml(input_xml=TEST_XML, camera_xml=camera_xml, output_xml_path="./assets/tmp_scene_w_camera.xml")


if __name__ == "__main__":
    main()