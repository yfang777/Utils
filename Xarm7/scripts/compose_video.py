#!/usr/bin/env python

import argparse
from pathlib import Path

import av
import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compose a directory of JPGs into an MP4 video using PyAV."
    )
    parser.add_argument(
        "--fig_dir",
        type=str,
        required=True,
        help="Directory containing JPG frames (e.g. 000000.jpg, 000001.jpg, ...).",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Output MP4 video path.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for the output video (default: 30).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    fig_dir = Path(args.fig_dir)
    video_path = Path(args.video_path)
    fps = args.fps

    if not fig_dir.is_dir():
        raise FileNotFoundError(f"Figure dir does not exist or is not a directory: {fig_dir}")

    # Collect JPG files and sort them
    frame_paths = sorted(
        p for p in fig_dir.iterdir()
        if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg"]
    )

    if not frame_paths:
        raise RuntimeError(f"No JPG files found in {fig_dir}")

    print(f"Found {len(frame_paths)} frames in {fig_dir}")

    # Read first frame to get size
    first_img = cv2.imread(str(frame_paths[0]))
    if first_img is None:
        raise RuntimeError(f"Failed to read first image: {frame_paths[0]}")

    height, width, _ = first_img.shape
    print(f"Frame size: {width}x{height}, FPS: {fps}")

    # Open output container
    container = av.open(str(video_path), mode="w")

    # Create a video stream (H.264 inside MP4)
    stream = container.add_stream("h264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"  # standard for H.264

    # Encode frames
    for i, frame_path in enumerate(frame_paths):
        img = cv2.imread(str(frame_path))
        if img is None:
            print(f"[WARN] Skipping unreadable image: {frame_path}")
            continue

        # Create VideoFrame (note: OpenCV is BGR, tell PyAV that)
        frame = av.VideoFrame.from_ndarray(img, format="bgr24")
        for packet in stream.encode(frame):
            container.mux(packet)

        if (i + 1) % 50 == 0:
            print(f"Encoded {i + 1}/{len(frame_paths)} frames...")

    # Flush encoder
    for packet in stream.encode():
        container.mux(packet)

    container.close()
    print(f"Saved video to: {video_path}")


if __name__ == "__main__":
    main()
