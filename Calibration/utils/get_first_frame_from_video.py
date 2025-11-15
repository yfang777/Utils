#!/usr/bin/env python3
import argparse
import av
import cv2
import sys

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True, help="Path to input .mp4")
    parser.add_argument("--output_file_path", type=str, required=True, help="Destination .jpg file")
    return parser.parse_args()

def main():
    args = get_args()

    container = av.open(args.video_path)

    frame_array = None
    for frame in container.decode(video=0):
        frame_array = frame.to_ndarray(format="bgr24")
        break
    container.close()

    cv2.imwrite(args.output_file_path, frame_array)
    print(f"[OK] Saved first frame to {args.output_file_path}")

if __name__ == "__main__":
    main()
