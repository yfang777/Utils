#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

# Your KeypointMapper + helpers must be importable from matcher.py
from matcher import KeypointMapper, compute_feats, compute_feats_batched

# ----------------------------
# Utils
# ----------------------------


def load_json_points(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # ensure ints
    kps = {str(k): {"x": int(v["x"]), "y": int(v["y"])} for k, v in data.items()}
    return kps

def foreground_mask_from_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """Foreground = any pixel that is not exactly [0,0,0]."""
    return np.any(img_bgr != 0, axis=2)  # (H, W) bool

# ----------------------------
# PCA (SVD-based; no sklearn needed)
# ----------------------------

def pca_fit(X: np.ndarray, n_components: int = 3):
    """
    X: (N, C) data matrix (float32/float64)
    returns dict with mean (C,), components (C, n), explained_var_ratio (n,)
    """
    X = np.asarray(X, dtype=np.float32)
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    # Economy SVD on (N x C)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)  # Xc = U S Vt
    # components are columns of V, i.e., rows of Vt
    components = Vt[:n_components].T  # (C, n)
    # explained variance per component
    n_samples = X.shape[0]
    explained_var = (S[:n_components] ** 2) / max(n_samples - 1, 1)
    total_var = (S ** 2).sum() / max(n_samples - 1, 1)
    explained_ratio = explained_var / (total_var + 1e-12)
    return {
        "mean": mean.reshape(-1),
        "components": components,
        "ratio": explained_ratio,
    }

def pca_transform(X: np.ndarray, mean: np.ndarray, components: np.ndarray):
    """
    X: (N, C); mean: (C,); components: (C, n)
    returns scores: (N, n)
    """
    Xc = X - mean.reshape(1, -1)
    return Xc @ components  # (N, n)

def pca_colorize_from_feats(feats_CHW: np.ndarray,
                            mask_HW: np.ndarray,
                            pca_fit_dict=None,
                            ref_min=None, ref_max=None):
    """
    feats_CHW: np array (C, H, W), float32
    mask_HW: bool (H, W) foreground mask; background stays black
    If pca_fit_dict is None -> fit PCA on masked pixels (reference pass), compute ref_min/ref_max on scores.
    If provided (target pass) -> use same mean/components and SAME ref_min/ref_max to normalize.

    Returns:
      color_bgr: np.uint8 (H, W, 3), background = black
      fit_dict, ref_min(3,), ref_max(3,)
    """
    C, H, W = feats_CHW.shape
    mask = mask_HW.astype(bool)


    F = feats_CHW.reshape(C, -1).T  # (H*W, C)
    idx = np.flatnonzero(mask.reshape(-1))
    X = F[idx]  # (N, C)

    if pca_fit_dict is None:
        fit = pca_fit(X, n_components=3)
        mean = fit["mean"]
        comps = fit["components"]
        ratio = fit["ratio"]
        scores = pca_transform(X, mean, comps)  # (N, 3)

        # min/max from REF foreground scores (per-channel)
        s_min = scores.min(axis=0)
        s_max = scores.max(axis=0)
    else:
        fit = pca_fit_dict
        mean = fit["mean"]
        comps = fit["components"]
        ratio = fit["ratio"]
        s_min = ref_min
        s_max = ref_max
        scores = pca_transform(X, mean, comps)

    # normalize to [0,1] using REF min/max to keep scale consistent
    denom = np.maximum(s_max - s_min, 1e-12)
    scores01 = (scores - s_min) / denom
    # weight each channel by explained variance ratio (emphasize PC with higher info)
    scores01 *= ratio.reshape(1, 3)

    # map to [0,255]
    scores255 = np.clip(scores01 * 255.0, 0, 255).astype(np.uint8)

    # place back into full image
    color = np.zeros((H * W, 3), dtype=np.uint8)
    color[idx] = scores255
    color = color.reshape(H, W, 3)

    # OpenCV uses BGR; our channels are arbitrary PCA axes. Keep as BGR for consistency.
    color_bgr = color  # treat 3 PCA channels as B,G,R in order

    # background stays black (already zero)
    return color_bgr, fit, s_min, s_max

# ----------------------------
# Viz
# ----------------------------

def draw_correspondences_side_by_side(color_ref_bgr, color_tgt_bgr,
                                      ref_kps, mapped_kps, out_path):
    """
    Concatenate images horizontally (no gap) and draw **red** lines from ref→tgt matches.
    """
    h, w = color_ref_bgr.shape[:2]
    canvas = np.concatenate([color_ref_bgr, color_tgt_bgr], axis=1)  # (H, 2W, 3)

    r = 4
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX

    for kid in sorted(ref_kps.keys(), key=lambda x: int(x) if x.isdigit() else x):
        u0, v0 = int(ref_kps[kid]["x"]), int(ref_kps[kid]["y"])
        u1, v1 = map(int, mapped_kps[kid])  # [u, v]
        p_left = (u0, v0)
        p_right = (u1 + w, v1)
        color = tuple(int(c) for c in np.random.randint(0, 255, size=3))
        # points
        cv2.circle(canvas, p_left, r, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, p_right, r, color, -1, lineType=cv2.LINE_AA)

        # labels
        cv2.putText(canvas, str(kid), (p_left[0] + 6, p_left[1] - 6),
                    font, 0.5, color, 1, cv2.LINE_AA)
        cv2.putText(canvas, str(kid), (p_right[0] + 6, p_right[1] - 6),
                    font, 0.5, color, 1, cv2.LINE_AA)

        # connecting line
        cv2.line(canvas, p_left, p_right, color, thickness, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    print(f"[ok] Saved visualization to: {out_path}")

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Map keypoints with DINO features, colorize ref/target via PCA on ref feats (bg=black), and draw red correspondences."
    )
    ap.add_argument("--ref_img", required=True, help="Path to reference image (640x480, jpg/png).")
    ap.add_argument("--ref_json", required=True, help="Path to reference keypoints JSON.")
    ap.add_argument("--tgt_img", required=True, help="Path to target image (640x480, jpg/png).")
    ap.add_argument("--model_location", required=True, help="Local repo dir for torch.hub.load.")
    ap.add_argument("--weight_path", required=True, help="Path to DINO weights.")
    ap.add_argument("--out", default=None, help="Output image (default: <tgt>_pca_corr.png).")
    args = ap.parse_args()

    ref_path = Path(args.ref_img)
    tgt_path = Path(args.tgt_img)
    json_path = Path(args.ref_json)

    # I/O
    ref_bgr = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
    tgt_bgr = cv2.imread(str(tgt_path), cv2.IMREAD_COLOR)

    ref_kps = load_json_points(json_path)

    # Init mapper and compute mapped kps
    mapper = KeypointMapper(args.model_location, args.weight_path)
    mapper.set_ref_image(ref_bgr, ref_kps)
    mapped_list = mapper.process_batch(np.expand_dims(tgt_bgr, axis=0))  # length 1
    mapped_kps = mapped_list[0]  # dict: kid -> np.array([u, v])

    # --- PCA colorization (bg=000 excluded from fit) ---
    feats_ref = mapper.feats_ref.detach().float().cpu().numpy()   # (C,H,W)
    feats_tgt = compute_feats_batched(mapper.model,
                                      np.expand_dims(tgt_bgr, 0))[0].detach().float().cpu().numpy()

    mask_ref = foreground_mask_from_bgr(ref_bgr)   # True for foreground
    mask_tgt = foreground_mask_from_bgr(tgt_bgr)   # True for foreground

    # Fit on REF foreground, then apply to both ref and target.
    color_ref, fit, s_min, s_max = pca_colorize_from_feats(feats_ref, mask_ref,
                                                           pca_fit_dict=None,
                                                           ref_min=None, ref_max=None)
    color_tgt, _,  _,  _  = pca_colorize_from_feats(feats_tgt, mask_tgt,
                                                    pca_fit_dict=fit,
                                                    ref_min=s_min, ref_max=s_max)

    # Compose output with correspondences
    out_path = Path(args.out) if args.out else tgt_path.with_name(tgt_path.stem + "_pca_corr.png")
    draw_correspondences_side_by_side(color_ref, color_tgt, ref_kps, mapped_kps, out_path)

    # (Optional) print mapped coords
    print("\nMapped keypoints (target image coords):")
    for kid in sorted(mapped_kps.keys(), key=lambda x: int(x) if x.isdigit() else x):
        u, v = map(int, mapped_kps[kid])
        print(f'  "{kid}": {{"x": {u}, "y": {v}}}')

if __name__ == "__main__":
    main()
