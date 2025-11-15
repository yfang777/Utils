dependencies:
    pip install "numpy<2.0" opencv-python==4.5.4.60 pyrealsense opencv-contrib-python==4.10.0.84 av mujoco einops

calibrator.py
* Realsense Calibrator
* Usage:
    python calibrator.py --rs_id 235422302222


reset.py
* reset the camera superparameter
* Usage:

overlay_mj_real.py
* Usage
    python overlay_mj_real.py \
        --xml "./assets/tmp_scene_w_camera.xml" \
        --img_path "./assets/mj_videoAndimg_for_overlay/episode_000000.jpg" \
        --rs_id 235422302222 \
        --rs_params_path "./assets/calib_params/cam_params.pkl"

    python overlay_MjToReal.py \
        --xml "./assets/tmp_scene_w_camera.xml" \
        --img_path "./assets/mujoco_imgs/scene_w_board.jpg" \
        --rs_id 235422302222 \
        --rs_params_path "./assets/calib_params/cam_params.pkl"

utils.get_first_frame_from_video.py
* Usage
    python utils.get_first_frame_from_video.py \
        --video_path "./assets/mj_videoAndimg_for_overlay/episode_000000.mp4" \
        --output_file_path "./assets/mj_videoAndimg_for_overlay/episode_000000.jpg"

utils.get_render_from_mujoco.py
* Usage
    python utils/get_render_from_mujoco.py \
        --xml "./assets/tmp_scene_w_camera.xml" \
        --camera_name "camera_235422302222" \
        --img_path "./data/mujoco_imgs/scene_w_camera.jpg"
