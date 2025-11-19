#!/usr/bin/env python3
import argparse
from pathlib import Path

import av
import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Overlay two videos with FPS downsampling and synced duration."
    )
    parser.add_argument("--video1", type=str, required=True)
    parser.add_argument("--video2", type=str, required=True)
    parser.add_argument("--output", type=str, default="overlay_output.mp4")
    parser.add_argument("--alpha", type=float, default=0.5)
    return parser.parse_args()


def get_fps(stream):
    """Return an integer FPS for PyAV stream."""
    if stream.average_rate is not None:
        return float(stream.average_rate)
    if stream.time_base:
        return 1.0 / stream.time_base
    return 30.0  # fallback


def main():
    args = parse_args()
    v1_path, v2_path = Path(args.video1), Path(args.video2)

    if not v1_path.is_file() or not v2_path.is_file():
        raise FileNotFoundError("One of the input videos does not exist.")

    container1 = av.open(str(v1_path))
    container2 = av.open(str(v2_path))

    stream1 = container1.streams.video[0]
    stream2 = container2.streams.video[0]

    fps1 = get_fps(stream1)
    fps2 = get_fps(stream2)
    out_fps = min(fps1, fps2)

    # duration (seconds)
    dur1 = float(stream1.duration * stream1.time_base) if stream1.duration else 1e9
    dur2 = float(stream2.duration * stream2.time_base) if stream2.duration else 1e9
    overlay_duration = min(dur1, dur2)

    # Geometry: resize video2 to match video1
    width = stream1.codec_context.width
    height = stream1.codec_context.height

    print("=== Video Info ===")
    print(f"Video1: fps={fps1:.2f}, duration={dur1:.2f}s")
    print(f"Video2: fps={fps2:.2f}, duration={dur2:.2f}s")
    print(f"Output fps = {out_fps}")
    print(f"Overlay duration = {overlay_duration:.2f}s")

    # Output
    out = av.open(args.output, "w")
    out_stream = out.add_stream("h264", rate=int(out_fps))
    out_stream.width = width
    out_stream.height = height
    out_stream.pix_fmt = "yuv420p"

    # Convert alpha
    alpha = float(np.clip(args.alpha, 0, 1))
    beta = 1 - alpha

    # Create iterators
    frames1 = container1.decode(stream1)
    frames2 = container2.decode(stream2)

    # Downsampling counters
    step1 = fps1 / out_fps
    step2 = fps2 / out_fps
    next_f1 = 0
    next_f2 = 0
    idx1 = idx2 = 0

    t = 0.0
    dt = 1.0 / out_fps
    out_count = 0

    print("=== Start overlay ===")

    f1 = next(frames1, None)
    f2 = next(frames2, None)

    while f1 is not None and f2 is not None and t <= overlay_duration:

        # --- skip frames for video1 ---
        while idx1 < next_f1:
            f1 = next(frames1, None)
            idx1 += 1
            if f1 is None:
                break

        # --- skip frames for video2 ---
        while idx2 < next_f2:
            f2 = next(frames2, None)
            idx2 += 1
            if f2 is None:
                break

        if f1 is None or f2 is None:
            break

        next_f1 += step1
        next_f2 += step2

        # convert frames to BGR
        img1 = f1.to_ndarray(format="bgr24")
        img2 = f2.to_ndarray(format="bgr24")
        img2 = cv2.resize(img2, (width, height))

        # overlay
        blended = cv2.addWeighted(img1, alpha, img2, beta, 0)

        # write frame
        out_frame = av.VideoFrame.from_ndarray(blended, format="bgr24")
        packet = out_stream.encode(out_frame)
        if packet:
            out.mux(packet)

        out_count += 1
        if out_count % 50 == 0:
            print(f"Processed {out_count} frames...")

        t += dt

    # flush encoder
    packet = out_stream.encode(None)
    if packet:
        out.mux(packet)

    out.close()
    container1.close()
    container2.close()

    print(f"[DONE] {out_count} frames written → {args.output}")


if __name__ == "__main__":
    main()
