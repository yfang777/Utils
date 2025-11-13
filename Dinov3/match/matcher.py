#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from typing import Dict, Tuple, List

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm import tqdm


# ----------------------------
# Config
# ----------------------------


MODEL_NAME = "dinov3_vitl16"

N_LAYERS       = 24
DINO_INPUT_WH  = (768, 768)
IMAGENET_MEAN  = (0.485, 0.456, 0.406)
IMAGENET_STD   = (0.229, 0.224, 0.225)



@torch.no_grad()
def load_dino_model(model_location, weight_path):
    model = torch.hub.load(
        repo_or_dir=model_location,
        model=MODEL_NAME,
        source="local",
        weights=weight_path
    )
    model.eval().cuda()
    return model

def frame_to_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    img_sq = pil_img.resize(DINO_INPUT_WH, Image.BICUBIC)
    img_t = TF.to_tensor(img_sq)
    img_t = TF.normalize(img_t, IMAGENET_MEAN, IMAGENET_STD)
    return img_t

@torch.no_grad()
def extract_feats(model, img_bgr_uint8) -> torch.Tensor:
    """
        Return: features (1, C, Hf, Wf)"""
    img_tensor = frame_to_tensor(img_bgr_uint8).unsqueeze(0).cuda()
    # img_tensor should be in rgb
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        
        feats = model.get_intermediate_layers(
            img_tensor, n=[N_LAYERS - 1], reshape=True, norm=True
        ) # [1, 1, 1024, 48, 48] 
    
    return feats[-1]

def interpolate_feats(feats, img_size=(640, 480)):
    W_img, H_img = img_size
    feat_interpolate = F.interpolate(
        feats,               # (1, C, Hf, Wf)
        size=(H_img, W_img),
        mode="bilinear",
        align_corners=True
    )
    return feat_interpolate


def compute_feats(model, img_bgr):
    '''
        img_bgr: cv2 image BGR order

        Return:
            feats: [C, W, H] (1024, 640, 480)
    '''
    assert img_bgr.shape[1] == 640 and  img_bgr.shape[0] == 480

    img_size = (img_bgr.shape[1], img_bgr.shape[0]) # (480, 640, 3)
    feats = extract_feats(model, img_bgr) #  (1, 1024, 48, 48) ;feat size (48, 48), patch size (16, 16)
    feats = interpolate_feats(feats, img_size)[0]  #  (1024, 768, 768)
    feats = F.normalize(feats, dim=0, p=2)
    return feats

@torch.no_grad()
def calc_sim_map(query_vec, target_feats):
    '''
        query_vec: [C]
        target_feats: [C, H, W] 

        Return: sim_map [h, w]
    '''

    sim_map = torch.einsum('c,chw->hw', query_vec, target_feats).clamp_(-1, 1)
    sim_map = sim_map.detach().float().cpu().numpy()
    return sim_map


@torch.no_grad()
def extract_feats_batched(model, img_bgr_batch) -> torch.Tensor:
    """
        img_bgr_batch: (N, H, W, 3)
        Return: features (N, C, Hf, Wf)"""
    


    batch_tensors = torch.stack([frame_to_tensor(f) for f in img_bgr_batch], dim=0).cuda()  # [1, 3, 768, 768]


    with torch.autocast(device_type="cuda", dtype=torch.float16):
        
        feats = model.get_intermediate_layers(
            batch_tensors, n=[N_LAYERS - 1], reshape=True, norm=True
        )[0] # [1, N, 1024, 48, 48] 
    return feats


@torch.no_grad()
def compute_feats_batched(model, img_bgr_batch) -> torch.Tensor:
    '''
        img_bgr_batch: [N, H, W, 3]
        Return:
            feats: torch.Tensor (B, C, H, W)
    '''

    assert img_bgr_batch.shape[2] == 640 and  img_bgr_batch.shape[1] == 480

    img_size = (img_bgr_batch.shape[2], img_bgr_batch.shape[1]) # (N, 480, 640, 3)
    feats = extract_feats_batched(model, img_bgr_batch) #  (N, 1024, 48, 48) ;feat size (48, 48), patch size (16, 16)
    feats = interpolate_feats(feats, img_size)  #  (N, 1024, 768, 768)
    feats = F.normalize(feats, dim=1, p=2)

    return feats

@torch.no_grad()
def calc_sim_map_batched(query_vec, target_feats_batch):
    '''
        query_vec: [C]
        target_feats: [C, H, W] 

        Return: sim_map [h, w]
    '''

    sim_map = torch.einsum('c,nchw->nhw', query_vec, target_feats_batch).clamp_(-1, 1)
    sim_map = sim_map.detach().float().cpu().numpy()
    return sim_map

def get_weighted_point_v2(sim_map, threshold=0.99):
    '''
        sim_map: [H, W]

        Return:
            point: [2]
    '''
    H, W = sim_map.shape

    sim_norm = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min())

    mask = sim_norm >= threshold
    num_nonzero = np.count_nonzero(mask)

    weights = sim_norm * mask
    weights_sum = np.sum(weights)

    v_idx, u_idx = np.nonzero(mask)         # coordinates of kept pixels
    w = sim_norm[mask]                           # weights on kept pixels
    w_sum = w.sum()

    u_weighted = (u_idx * np.exp(w)).sum() / np.exp(w).sum()
    v_weighted = (v_idx * np.exp(w)).sum() / np.exp(w).sum()

    return np.array([int(v_weighted), int(u_weighted)])



# Interpolate in Image space
class KeypointMapper:
    def __init__(self, model_location, weight_path):
        self.model = load_dino_model(model_location, weight_path)

    def set_ref_image(self, image, keypoints):
        '''
            img: BGR image cv2.imread
            keypoints: {'1': (u, v), '2': ...}
        '''

        
        self.feats_ref = compute_feats(self.model, image)
        self.kps_ref = keypoints

    

    def process_batch(self, img_bgr_batch):
        '''
            img_bgr_batch: [N, H, W, C]
        '''

        N = img_bgr_batch.shape[0]
        frames_mapped_kps = [dict() for _ in range(N)]

        kp_ids = [kid for kid in self.kps_ref]        

        feats_tgt_batch = compute_feats_batched(self.model, img_bgr_batch)  # (N, C, Hf, Wf)
        
        for kid in kp_ids:
            u, v = int(self.kps_ref[kid]["x"]), int(self.kps_ref[kid]["y"])

            ref_vec = self.feats_ref[:, v, u]

            sim_map_batch = calc_sim_map_batched(ref_vec, feats_tgt_batch)

            for i in range(N):
                v_i, u_i = get_weighted_point_v2(sim_map_batch[i])

                frames_mapped_kps[i][kid] = np.array([int(u_i), int(v_i)])

        return frames_mapped_kps
