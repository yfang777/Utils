#!/usr/bin/env python3

import argparse
import os
import cv2


def overlay_images(img1_path, img2_path, output_path):
    # Load images (as BGR)
    img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
    img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    alpha = 0.5
    beta = 1.0 - alpha
    overlay = cv2.addWeighted(img1, alpha, img2, beta, 0.0)
    
    cv2.imwrite(output_path, overlay)



def get_args():
    parser = argparse.ArgumentParser(
        description="Overlay two images (image2 resized to match image1)."
    )
    parser.add_argument("--image1_path", type=str, help="Path to the first image")
    parser.add_argument("--image2_path", type=str, help="Path to the second image")
    parser.add_argument("--output_path", type=str)
    return parser.parse_args()


def main():
    args = get_args()
    overlay_images(args.image1_path, args.image2_path, args.output_path)


if __name__ == "__main__":
    main()
