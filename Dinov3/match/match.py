#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from matcher import KeypointMapper


def load_json_points(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    kps = {}
    for k, v in data.items():
        kps[str(k)] = {"x": int(v["x"]), "y": int(v["y"])}
    return kps

def draw_correspondences_side_by_side(ref_bgr, tgt_bgr, ref_kps, mapped_kps, out_path):
    """
    Concatenate images horizontally (no gap) and draw colored lines from ref→tgt matches.
    Each keypoint pair uses a different random color.
    """
    h, w = ref_bgr.shape[:2]
    canvas = np.concatenate([ref_bgr, tgt_bgr], axis=1)  # (H, 2W, 3)

    r = 4
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX

    random.seed(42)  # for reproducibility; remove or change if you want fully random colors

    for kid in sorted(ref_kps.keys(), key=lambda x: int(x) if x.isdigit() else x):
        # random BGR color
        color = tuple(int(c) for c in np.random.randint(0, 255, size=3))

        u0, v0 = int(ref_kps[kid]["x"]), int(ref_kps[kid]["y"])
        u1, v1 = map(int, mapped_kps[kid])  # [u, v]
        p_left = (u0, v0)
        p_right = (u1 + w, v1)

        # draw points and line
        cv2.circle(canvas, p_left, r, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, p_right, r, color, -1, lineType=cv2.LINE_AA)
        cv2.line(canvas, p_left, p_right, color, thickness, cv2.LINE_AA)

        # draw label
        cv2.putText(canvas, str(kid), (p_left[0] + 6, p_left[1] - 6), font, 0.5, color, 1, cv2.LINE_AA)
        cv2.putText(canvas, str(kid), (p_right[0] + 6, p_right[1] - 6), font, 0.5, color, 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    print(f"[ok] Saved visualization to: {out_path}")

def main():
    ap = argparse.ArgumentParser(
        description="Map keypoints from a reference image to a target image with DINO features, then draw red correspondence lines on a side-by-side concat."
    )
    ap.add_argument("--ref_img", required=True, help="Path to reference image (640x480, jpg/png).")
    ap.add_argument("--ref_json", required=True, help="Path to reference keypoints JSON.")
    ap.add_argument("--tgt_img", required=True, help="Path to target image (640x480, jpg/png).")
    ap.add_argument("--model_location", required=True, help="Local repo dir passed to torch.hub.load (contains your DINO).")
    ap.add_argument("--weight_path", required=True, help="Path to DINO weights.")
    ap.add_argument("--out", default=None, help="Output image path (default: <tgt_img_stem>_correspondence.png).")
    args = ap.parse_args()

    ref_path = Path(args.ref_img)
    tgt_path = Path(args.tgt_img)
    json_path = Path(args.ref_json)

    # I/O
    ref_bgr = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
    tgt_bgr = cv2.imread(str(tgt_path), cv2.IMREAD_COLOR)
 
    ref_kps = load_json_points(json_path)

    # Init mapper and compute
    mapper = KeypointMapper(args.model_location, args.weight_path)
    mapper.set_ref_image(ref_bgr, ref_kps)

    mapped_list = mapper.process_batch(np.expand_dims(tgt_bgr, axis=0))  # length 1
    mapped_kps = mapped_list[0]  # dict: kid -> np.array([u, v])

    # Save visualization
    out_path = Path(args.out) if args.out else tgt_path.with_name(tgt_path.stem + "_correspondence.png")
    draw_correspondences_side_by_side(ref_bgr, tgt_bgr, ref_kps, mapped_kps, out_path)

    # Optional: print the mapped coordinates
    print("\nMapped keypoints (target image coords):")
    for kid in sorted(mapped_kps.keys(), key=lambda x: int(x) if x.isdigit() else x):
        u, v = map(int, mapped_kps[kid])
        print(f'  "{kid}": {{"x": {u}, "y": {v}}}')

if __name__ == "__main__":
    main()
