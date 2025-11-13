#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser(
        description="Click points on a 640x480 image and save them as JSON."
    )
    p.add_argument("--image", required=True, help="Path to input image (jpg/png).")
    p.add_argument("--out", default=None, help="Output JSON file (default: image basename + _points.json).")
    p.add_argument("--title", default="Click points (left-click to add, 'u' undo, 's' save, 'q' quit)",
                   help="Window title / on-screen instructions.")
    args = p.parse_args()

    img_path = Path(args.image)
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    if not (w == 640 and h == 480):
        print(f"Error: expected image size 640x480 (W×H), got {w}x{h}", file=sys.stderr)
        raise ValueError

    out_path = Path(args.out)

    points = []
    display = img.copy()

    # Drawing helpers
    def redraw():
        nonlocal display
        display = img.copy()
        for i, (px, py) in enumerate(points, start=1):
            cv2.circle(display, (px, py), 4, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
            cv2.putText(display, str(i), (px + 6, py - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(display, args.title, (8, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                    thickness=1, lineType=cv2.LINE_AA)

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((int(x), int(y)))
            redraw()

    win = "point_picker"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    redraw()

    while True:
        cv2.imshow(win, display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('u'):  # undo last
            if points:
                points.pop()
                redraw()
        elif key == ord('q') or key == 27:  # 'q' or ESC quits without saving
           break
    cv2.destroyAllWindows()

    # Build { "1": {"x":..., "y":...}, "2": {...}, ... }
    out_obj = {str(i + 1): {"x": int(px), "y": int(py)} for i, (px, py) in enumerate(points)}

    # Save JSON
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2)
    print(f"Saved {len(points)} points to {out_path}")


if __name__ == "__main__":
    main()
