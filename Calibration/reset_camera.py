import pyrealsense2 as rs

def reset_realsense_to_default_simple():
    ctx = rs.context()
    serials = [dev.get_info(rs.camera_info.serial_number) for dev in ctx.query_devices()]
    print(f"Found {len(serials)} devices:", serials)

    for serial in serials:
        print(f"\nResetting camera {serial} ...")
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        pipe = rs.pipeline()
        profile = pipe.start(cfg)
        dev = profile.get_device()

        # ---------- Reset color sensor ----------
        try:
            color_sensor = dev.first_color_sensor()
            print("  Resetting color sensor...")
            for opt in color_sensor.get_supported_options():
                try:
                    rng = color_sensor.get_option_range(opt)
                    val = getattr(rng, "default", getattr(rng, "def_", None))

                    print("rng", rng)
                    print("val", val)
                    if val is not None:
                        color_sensor.set_option(opt, float(val))
                except Exception:
                    pass


            # Disable auto modes
            if color_sensor.supports(rs.option.enable_auto_exposure):
                color_sensor.set_option(rs.option.enable_auto_exposure, 0)
            if color_sensor.supports(rs.option.enable_auto_white_balance):
                color_sensor.set_option(rs.option.enable_auto_white_balance, 0)

            color_sensor.set_option(rs.option.exposure, 166)
            color_sensor.set_option(rs.option.white_balance, 5000)
            print("  ✅ Color sensor reset complete.")
        except Exception as e:
            print(f"  ⚠️ Color sensor reset failed: {e}")

        # ---------- Reset depth sensor ----------
        try:
            depth_sensor = dev.first_depth_sensor()
            print("  Resetting depth sensor...")
            for opt in depth_sensor.get_supported_options():
                try:
                    rng = depth_sensor.get_option_range(opt)
                    val = getattr(rng, "default", getattr(rng, "def_", None))
                    if val is not None:
                        depth_sensor.set_option(opt, float(val))
                except Exception:
                    pass
            print("  ✅ Depth sensor reset complete.")
        except Exception as e:
            print(f"  ⚠️ Depth sensor reset failed: {e}")

        pipe.stop()
    print("\nAll cameras reset to default successfully.")






if __name__ == "__main__":
    reset_realsense_to_default_simple()
