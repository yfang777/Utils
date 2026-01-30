import pyzed.sl as sl
import numpy as np
import open3d as o3d
import time

def main():
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    # init_params.coordinate_units = sl.UNIT.METER
    # Lower the min depth if you are testing on a desk
    init_params.depth_minimum_distance = 0.2

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED")
        return

    # --- FIX 1: WARM UP ---
    print("Warming up sensors...")
    for i in range(30): # Grab 30 frames to let auto-exposure settle
        zed.grab()
    
    # --- FIX 2: VALIDATION LOOP ---
    zed_pc = sl.Mat()
    points = np.array([])
    
    print("Searching for a valid depth frame...")
    for _ in range(50): # Try for up to 50 frames
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_measure(zed_pc, sl.MEASURE.XYZRGBA)
            data = zed_pc.get_data().reshape(-1, 4)
            
            # Check for non-NaN points
            mask = ~np.isnan(data[:, :3]).any(axis=1)
            valid_data = data[mask]
            
            if len(valid_data) > 1000: # We want at least 1000 points
                print(f"Found {len(valid_data)} valid points!")
                points = valid_data[:, :3]
                
                # Unpack colors
                packed_colors = valid_data[:, 3].view(np.uint32)
                r = ((packed_colors >> 0) & 0xFF) / 255.0
                g = ((packed_colors >> 8) & 0xFF) / 255.0
                b = ((packed_colors >> 16) & 0xFF) / 255.0
                colors = np.stack((r, g, b), axis=-1)
                break
        time.sleep(0.1)

    if len(points) == 0:
        print("Still no points. Check if the lens cap is off or if you are too close to an object.")
        zed.close()
        return

    # Visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Flip the visualization so it's right-side up
    pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    
    o3d.visualization.draw_geometries([pcd])
    zed.close()

if __name__ == "__main__":
    main()