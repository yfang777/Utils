#!/usr/bin/env python3
import argparse
from pathlib import Path

import av
import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Overlay two videos (alpha blend) and save the result using PyAV."
    )
    parser.add_argument(
        "--video1",
        type=str,
        required=True,
        help="Path to the first video (base layer).",
    )
    parser.add_argument(
        "--video2",
        type=str,
        required=True,
        help="Path to the second video (overlay layer).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="overlay_output.mp4",
        help="Path to the output overlaid video.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Blend weight for video1 (0.0–1.0). video2 gets (1 - alpha).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    v1_path = Path(args.video1)
    v2_path = Path(args.video2)

    if not v1_path.is_file():
        raise FileNotFoundError(f"video1 not found: {v1_path}")
    if not v2_path.is_file():
        raise FileNotFoundError(f"video2 not found: {v2_path}")

    # Open containers
    container1 = av.open(str(v1_path))
    container2 = av.open(str(v2_path))

    # Get first video streams
    stream1 = next(s for s in container1.streams if s.type == "video")
    stream2 = next(s for s in container2.streams if s.type == "video")

    # Use video1 properties as reference
    width = stream1.codec_context.width
    height = stream1.codec_context.height


    fps = 30

    alpha = float(args.alpha)
    alpha = max(0.0, min(1.0, alpha))
    beta = 1.0 - alpha

    # Output container
    out_container = av.open(args.output, mode="w")
    out_stream = out_container.add_stream("h264", rate=fps)
    out_stream.width = width
    out_stream.height = height
    out_stream.pix_fmt = "yuv420p"

    print(f"[info] Writing overlaid video to: {args.output}")
    print(f"[info] alpha={alpha:.2f}, beta={beta:.2f}, fps={fps:.2f}")

    frame_idx = 0

    # Decode frames from both videos; stop when one ends
    frames1 = container1.decode(stream1)
    frames2 = container2.decode(stream2)

    for f1, f2 in zip(frames1, frames2):
        # Convert to BGR numpy arrays
        img1 = f1.to_ndarray(format="bgr24")
        img2 = f2.to_ndarray(format="bgr24")


        # Alpha blend
        overlaid = cv2.addWeighted(img1, alpha, img2, beta, 0.0)

        # Encode and mux
        out_frame = av.VideoFrame.from_ndarray(overlaid, format="bgr24")
        packet = out_stream.encode(out_frame)
        if packet:
            out_container.mux(packet)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"[info] processed {frame_idx} frames...", end="\r", flush=True)

    # Flush encoder
    packet = out_stream.encode(None)
    if packet:
        out_container.mux(packet)

    out_container.close()
    container1.close()
    container2.close()

    print(f"\n[done] Total frames written: {frame_idx}")


if __name__ == "__main__":
    main()
