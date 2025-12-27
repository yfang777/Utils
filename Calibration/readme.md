dependencies:
    pip install "numpy<2.0" opencv-python==4.5.4.60 pyrealsense opencv-contrib-python==4.10.0.84 av mujoco einops

calibrator.py
* python calibrator.py --rs_id "235422302222"

reset.py
* reset the camera superparameter

```
python rgb_match.py \
    --xml "./assets/tmp_scene_w_camera.xml" \
    --rs_id 235422302222 \
    --rs_params_path "./assets/calib_params/cam_params.pkl"

python rgb_match_dynamic.py \
    --xml "./assets/tmp_scene_w_camera.xml" \
    --rs_id 235422302222 \
    --rs_params_path "./assets/calib_params/cam_params.pkl"

python pointcloud_match.py \
    --input_xml_path "./assets/tmp_scene_w_camera.xml" \
    --rs_id "235422302222" \
    --input_camera_path "./assets/calib_params/cam_params.pkl"

python pointcloud_match_dynamic.py \
    --xml "./assets/tmp_scene_w_camera.xml" \
    --rs_id "235422302222" \
    --rs_params_path "./assets/calib_params/cam_params.pkl"

```

utils.get_first_frame_from_video.py
* Usage
    python -m utils.get_first_frame_from_video \
        --video_path "/home/yuan/codegen/Utils/Xarm7/assets/replay_trajs/v5/episode_000001.mp4" \
        --output_file_path "./data/mujoco_imgs/v5/episode_000001.jpg"

utils.get_render_from_mujoco.py
* Usage
    python utils/get_render_from_mujoco.py \
        --xml "/home/yuan/codegen/Utils/Calibration/assets/ufactory_xarm7/scene.xml" \
        --camera_name "camera_235422302222" \
        --img_path "./data/mujoco_imgs/xarm7.jpg"

utils.overlay_two_images.py
* Usage
    python utils/overlay_two_images.py \
        --image1_path "/home/yuan/codegen/Utils/Calibration/data/mujoco_imgs/v1/episode_000001.jpg" \
        --image2_path "/home/yuan/codegen/Utils/Calibration/data/mujoco_imgs/v2/episode_000001.jpg" \
        --output_path "/home/yuan/codegen/Utils/Calibration/data/mujoco_imgs/overlay.jpg"

