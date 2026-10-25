"""Fine-tune ViTPose+ small (MoE backbone, COCO-17 head) on UPAR-Pose train data.

Target layout is PoseTrack18-17 slots. We train a lightweight pose-specific head that
remaps/renames channels, while routing through a frozen-then-unfrozen MoE backbone.

Design decisions (rationale):
- Base: usyd-community/vitpose-plus-small. MoE experts: 0=COCO, 1=AiC, 2=MPII, 3=AP-10K,
  4=APT-36K, 5=COCO-WholeBody (per HF transformers vitpose docs). The single shared
  decoder head (conv 17ch) means ALL experts output COCO-ordered channels.
- GT layout order (UPAR): [nose, head_bottom, head_top, l_ear, r_ear, l_sho, r_sho,
  l_elb, r_elb, l_wri, r_wri, l_hip, r_hip, l_kne, r_kne, l_ank, r_ank].
- Since we fine-tune on UPAR data (allowed: organizer-provided training data), the head
  learns the PT18 slot semantics directly from data, fixing the head_bottom/head_top
  misalignment that the zero-shot baseline suffers (15.65px / 12.76px errors on slots 1/2).
- Channel remap from COCO order to PT18 slots (initialization prior):
    PT18 nose      <- COCO nose (0)
    PT18 head_bottom <- mean(eyes); head_top <- mean(ear,height-2*ear)
  Better: replace head with learned 1x1 conv from all 5 COCO head channels.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

DATA_ROOT = os.path.abspath(os.environ.get("UPAR_DATA", os.path.join(os.path.dirname(__file__), "..", "data")))
ANN_FILE = os.path.join(DATA_ROOT, "annotations/specialization/train_upar_all.json")
CACHE_ROOT = os.environ.get("UPAR_CACHE", "/tmp/upar_train_cache")

# PoseTrack18 slot order used by UPAR
PT18 = ["nose", "head_bottom", "head_top", "l_ear", "r_ear", "l_sho", "r_sho", "l_elb", "r_elb",
        "l_wri", "r_wri", "l_hip", "r_hip", "l_kne", "r_kne", "l_ank", "r_ank"]
# COCO order of the ViTPose+ head
COCO = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_sho", "r_sho", "l_elb", "r_elb",
        "l_wri", "r_wri", "l_hip", "r_hip", "l_kne", "r_kne", "l_ank", "r_ank"]
# map PT18 index -> COCO indices to average (None = no equivalent, predict from scratch)
PT18_FROM_COCO = [
    [0],        # nose <- nose
    [1, 2],     # head_bottom <- mean(l_eye, r_eye)  (eyes sit at neck/eye height; fine-tune refines)
    None,       # head_top <- no COCO equivalent (top of skull): learned
    [3],        # l_ear <- l_ear
    [4],        # r_ear <- r_ear
    [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16],
]

FLIP_PAIRS_PT18 = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]


def box_to_center_scale(box, out_w, out_h, normalize_factor=200.0, padding_factor=1.25):
    x, y, w, h = [float(v) for v in box[:4]]
    aspect = out_w / out_h
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
    if w > aspect * h:
        h = w / aspect
    elif w < aspect * h:
        w = h * aspect
    scale = np.array([w / normalize_factor, h / normalize_factor], dtype=np.float32)
    return center, scale * padding_factor


def warp_matrix(center, scale, out_w, out_h):
    size_input = center * 2.0
    size_dst = np.array([out_w, out_h], dtype=np.float32) - 1.0
    size_target = scale * 200.0
    scale_x = size_dst[0] / size_target[0]
    scale_y = size_dst[1] / size_target[1]
    matrix = np.zeros((2, 3), dtype=np.float32)
    matrix[0, 0] = scale_x
    matrix[0, 2] = scale_x * (-0.5 * size_input[0] + 0.5 * size_target[0])
    matrix[1, 1] = scale_y
    matrix[1, 2] = scale_y * (-0.5 * size_input[1] + 0.5 * size_target[1])
    return matrix


def warp_affine(image, matrix, out_h, out_w):
    bottom = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    inverse = np.linalg.inv(np.vstack([matrix, bottom]))[:2].astype(np.float32)
    ys, xs = np.meshgrid(np.arange(out_h, dtype=np.float32),
                         np.arange(out_w, dtype=np.float32), indexing="ij")
    src_x = inverse[0, 0] * xs + inverse[0, 1] * ys + inverse[0, 2]
    src_y = inverse[1, 0] * xs + inverse[1, 1] * ys + inverse[1, 2]
    height, width = image.shape[:2]
    x0 = np.floor(src_x).astype(np.int64); y0 = np.floor(src_y).astype(np.int64)
    x1, y1 = x0 + 1, y0 + 1
    wx, wy = src_x - x0, src_y - y0
    inside = (src_x >= 0) & (src_x <= width - 1) & (src_y >= 0) & (src_y <= height - 1)
    x0c, x1c = np.clip(x0, 0, width - 1), np.clip(x1, 0, width - 1)
    y0c, y1c = np.clip(y0, 0, height - 1), np.clip(y1, 0, height - 1)
    top = image[y0c, x0c] * (1 - wx)[..., None] + image[y0c, x1c] * wx[..., None]
    bot = image[y1c, x0c] * (1 - wx)[..., None] + image[y1c, x1c] * wx[..., None]
    return (top * (1 - wy)[..., None] + bot * wy[..., None]) * inside[..., None]


def transform_kps(kps, matrix, out_h=None, out_w=None, inverse=False):
    """kps: (N,3) -> transformed (N,3) with x,y mapped; v preserved."""
    kps = kps.copy()
    xy = kps[:, :2]
    if inverse:
        bottom = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        m = np.linalg.inv(np.vstack([matrix, bottom]))[:2].astype(np.float32)
    else:
        m = matrix
    ones = np.ones((len(xy), 1), dtype=np.float32)
    xy_h = np.concatenate([xy, ones], axis=1).T  # (3,N)
    out = m @ xy_h  # (2,N)
    kps[:, :2] = out.T
    return kps


class UPARDataset(torch.utils.data.Dataset):
    def __init__(self, ann_file, data_root, input_size=(256, 192), sigma=2.0,
                 augment=True, hemi_prob=0.3, hemi_offset=0.06, flip_prob=0.5,
                 scale_factor=0.3, rotate_deg=30, color_prob=0.5, mean=None, std=None,
                 cache_crops=False, cache_root=None, val_ids=None, is_val=False):
        self.ann = json.load(open(ann_file))
        self.imgs = {im["id"]: im for im in self.ann["images"]}
        self.data_root = data_root
        self.out_h, self.out_w = input_size  # (256,192)
        self.heatmap_size = (self.out_h // 4, self.out_w // 4)
        self.sigma = sigma
        self.augment = augment
        self.flip_prob = flip_prob
        self.scale_factor = scale_factor
        self.rotate_deg = rotate_deg
        self.color_prob = color_prob
        self.hemi_prob = hemi_prob
        self.hemi_offset = hemi_offset
        self.mean = np.array(mean if mean else [0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(std if std else [0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        if val_ids is not None:
            val_set = set(val_ids)
            self.items = [(a, self.imgs[a["image_id"]]) for a in self.ann["annotations"]
                          if (a["id"] in val_set) == is_val]
        else:
            self.items = [(a, self.imgs[a["image_id"]]) for a in self.ann["annotations"]]

    def __len__(self):
        return len(self.items)

    def _load_image(self, file_name):
        path = os.path.join(self.data_root, file_name)
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0

    def __getitem__(self, idx):
        a, im = self.items[idx]
        img = self._load_image(im["file_name"])
        kps = np.array(a["keypoints"], dtype=np.float32).reshape(-1, 3)
        bbox = a["bbox"]

        center, scale = box_to_center_scale(bbox, self.out_w, self.out_h)

        if self.augment:
            # random scale
            sf = np.clip(np.random.randn() * 0.1 + 1.0, 1 - self.scale_factor, 1 + self.scale_factor)
            scale = scale * sf
            # random rotate about center
            deg = (np.random.rand() * 2 - 1) * self.rotate_deg
            # build rotation+scale affine around center, then warp
            rad = math.radians(deg)
            cos, sin = math.cos(rad), math.sin(rad)
            # rotate keypoints about box center in original image coords first
            rc = center
            rot = np.array([[cos, -sin, rc[0] - cos * rc[0] + sin * rc[1]],
                            [sin, cos, rc[1] - sin * rc[0] - cos * rc[1]]], dtype=np.float32)
            # apply to kps for heatmap target only (image is rotated by warp)
            # we instead rotate the image and kps together via affine matrix composition
            mat = warp_matrix(center, scale, self.out_w, self.out_h)
            # full transform = mat @ rot  (rotate in source space, then warp)
            rot3 = np.vstack([rot, [0, 0, 1]])
            mat3 = np.vstack([mat, [0, 0, 1]])
            full = (mat3 @ rot3)
            crop = warp_affine(img, full[:2].astype(np.float32), self.out_h, self.out_w)
            kps_t = transform_kps(kps, full[:2].astype(np.float32))
            # random flip
            if np.random.rand() < self.flip_prob:
                crop = crop[:, ::-1]
                kps_t = kps_t.copy()
                kps_t[:, 0] = self.out_w - 1.0 - kps_t[:, 0]
                # swap left/right
                for a_, b_ in [(3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]:
                    kps_t[a_], kps_t[b_] = kps_t[b_].copy(), kps_t[a_].copy()
            # color jitter (brightness/contrast)
            if np.random.rand() < self.color_prob:
                alpha = 1.0 + (np.random.rand() * 0.4 - 0.2)
                beta = np.random.rand() * 0.2 - 0.1
                crop = crop * alpha + beta
        else:
            mat = warp_matrix(center, scale, self.out_w, self.out_h)
            crop = warp_affine(img, mat, self.out_h, self.out_w)
            kps_t = transform_kps(kps, mat)

        crop = np.clip(crop, 0, 1)
        # hemispheric missing modality augmentation: black out left or right half of BODY
        if self.augment and np.random.rand() < self.hemi_prob:
            side = np.random.choice(["l", "r"])
            # derive from expected positions: l half is x < nose x, r half is x > nose x
            nose_x = kps_t[0, 0] if kps_t[0, 2] > 0 else self.out_w / 2
            if side == "l":
                xs = np.arange(self.out_w) < nose_x - self.out_w * self.hemi_offset * 0
            else:
                xs = np.arange(self.out_w) > nose_x + self.out_w * self.hemi_offset * 0
            mask = np.tile(xs[None, :, None], (self.out_h, 1, 3)).astype(np.float32)
            crop = crop * (1 - mask)
            # leave kps labels unchanged (occlusion-style augmentation)

        tensor = crop.transpose(2, 0, 1)
        tensor = (tensor - self.mean) / self.std

        # heatmaps
        hm = np.zeros((17, *self.heatmap_size), dtype=np.float32)
        # UDP unbiased encode: continuous-center gaussians on the (N-1)/(n-1) stride grid,
        # matching the decoder's linear transform-back — removes the systematic 0-3px stretch
        # the naive /4.0 quantized encode suffered at crop edges.
        stride_y = (self.out_h - 1.0) / (self.heatmap_size[0] - 1.0)
        stride_x = (self.out_w - 1.0) / (self.heatmap_size[1] - 1.0)
        kps_hm = np.empty_like(kps_t[:, :2])
        kps_hm[:, 1] = kps_t[:, 1] / stride_y
        kps_hm[:, 0] = kps_t[:, 0] / stride_x
        for i in range(17):
            if kps_t[i, 2] > 0:
                x, y = kps_hm[i]
                xi, yi = int(x + 0.5), int(y + 0.5)
                r = int(3 * self.sigma)
                x0, x1 = max(0, xi - r), min(self.heatmap_size[1], xi + r + 1)
                y0, y1 = max(0, yi - r), min(self.heatmap_size[0], yi + r + 1)
                if x1 <= x0 or y1 <= y0:
                    continue
                yy, xx = np.mgrid[y0:y1, x0:x1]
                d2 = (xx - x) ** 2 + (yy - y) ** 2
                hm[i, y0:y1, x0:x1] = np.maximum(hm[i, y0:y1, x0:x1], np.exp(-d2 / (2 * self.sigma ** 2)))

        vis_mask = (kps_t[:, 2] > 0).astype(np.float32)
        return {
            "pixel_values": torch.from_numpy(tensor.astype(np.float32)),
            "heatmaps": torch.from_numpy(hm),
            "vis_mask": torch.from_numpy(vis_mask),
            "image_id": im["id"],
            "anno_id": a["id"],
        }


class RemapHead(nn.Module):
    """17->17 channel remap + depthwise refinement, NO normalization.

    Learned the hard way: BatchNorm/GroupNorm over random-init or near-identity features
    explodes activations (loss 1.8e4-1.4e5); heatmaps are already ~[0,1] so a plain
    conv stack initialized as the fixed COCO->PT18 channel remap is stable and starts
    at the zero-shot baseline's behavior. """
    def __init__(self, ch=17):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 1)
        self.dw = nn.Conv2d(ch, ch, 3, padding=1, groups=ch)
        self.conv2 = nn.Conv2d(ch, ch, 1)
        with torch.no_grad():
            self.conv1.weight.copy_(torch.eye(ch).view(ch, ch, 1, 1)); self.conv1.bias.zero_()
            self.dw.weight.zero_(); self.dw.bias.zero_()
            self.conv2.weight.zero_(); self.conv2.bias.zero_()
            for pt_i, coco_idx in enumerate(PT18_FROM_COCO):
                if coco_idx is not None:
                    for ci in coco_idx:
                        self.conv2.weight[pt_i, ci, 0, 0] = 1.0 / len(coco_idx)
            self.conv2.weight[2, 3, 0, 0] = 0.5   # head_top <- ear-mean proxy
            self.conv2.weight[2, 4, 0, 0] = 0.5

    def forward(self, coco_hm):
        x = F.relu(self.conv1(coco_hm))
        return self.conv2(x + self.dw(x))


class FinetuneViTPose(nn.Module):
    """ViTPose+ backbone routed to COCO expert (0) + PT18 remap head.

    The backbone output (COCO-ordered heatmaps) is the frozen feature prior; the remap
    head learns PT18 semantics. Optionally unfreeze backbone for full fine-tune.
    """
    def __init__(self, vitpose, remap_in=17, freeze_backbone=True):
        super().__init__()
        self.vitpose = vitpose
        self.remap = RemapHead(ch=remap_in)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.vitpose.parameters():
                p.requires_grad = False
            self.vitpose.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.vitpose.eval()
        return self

    def forward(self, pixel_values, dataset_index=None):
        if dataset_index is None:
            dataset_index = torch.zeros(pixel_values.shape[0], dtype=torch.int64,
                                        device=pixel_values.device)
        if self.freeze_backbone:
            with torch.no_grad():
                coco_hm = self.vitpose(pixel_values=pixel_values,
                                       dataset_index=dataset_index).heatmaps
        else:
            coco_hm = self.vitpose(pixel_values=pixel_values,
                                   dataset_index=dataset_index).heatmaps
        return self.remap(coco_hm)

