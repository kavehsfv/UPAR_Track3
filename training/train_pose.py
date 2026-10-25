"""Final mipnet_v2 training recipe used by the mipnet_v3 inference entry.

ViTPose+ Small -> learned PT18 remapping -> ProbPose-inspired probability
heatmaps and detached in-crop presence branch. Adapted ProbPose mechanisms:
OKS-shaped targets, Sparsemax temperature 0.5, expected-OKS loss with Sobel
smoothing 0.05, and OKS-kernel/Taylor decoding.

The box-noise augmentation is a local adaptation motivated by imperfect-box
robustness studied with MIPNet; it is NOT the MIPNet multi-instance network.
Defaults are the selected reduced dose: p=0.25; center shift up to 5% of
width/height; per-axis size perturbation up to 7.5%. Keypoints are unchanged
in source-image coordinates. Fifteen epochs with backbone unfreeze at epoch 4.

ProbPose: Purkrabek & Matas, CVPR 2025, arXiv:2412.02254.
Source studied: MiraPurkrabek/ProbPose_code; see THIRD_PARTY_NOTICES.md.
MIPNet source studied: rawalkhirodkar/MIPNet.
Inference in ../run.py adds three-scale flip testing and uses the convolved
peak confidence, rather than the raw-map score used in early experiments.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("tv", os.path.join(_HERE, "train_vitpose.py"))
tv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tv)

ANN_FILE = tv.ANN_FILE
DATA_ROOT = tv.DATA_ROOT
box_to_center_scale = tv.box_to_center_scale
RemapHead = tv.RemapHead
FinetuneViTPose = tv.FinetuneViTPose
UPARDataset = tv.UPARDataset
warp_matrix = tv.warp_matrix
warp_affine = tv.warp_affine
transform_kps = tv.transform_kps
# (train_vitpose exec'd as module "tv" above — a plain `from train_vitpose
# import` fails when sys.path lacks training/, e.g. during TorchScript export)
# flat helpers re-exported at module level in train_vitpose.py: warp_matrix,
# warp_affine, transform_kps, box_to_center_scale, RemapHead, UPARDataset.

# PT18 per-slot sigmas (eval_local.py SIGMAS order): nose .025, head_bottom
# .025, head_top .059, earL .059, earR .059, shoulderL .079, shoulderR .079,
# elbowL .072, elbowR .072, wristL .062, wristR .062, hipL .107, hipR .107,
# kneeL .087, kneeR .087, ankleL .089, ankleR .089
PT18_SIGMAS = np.array(
    [.025, .025, .059, .059, .059, .079, .079, .072, .072, .062, .062,
     .107, .107, .087, .087, .089, .089], dtype=np.float32)

HM_H, HM_W = 64, 48          # heatmap grid
OUT_W, OUT_H = 192, 256      # crop size
BBOX_AREA = math.sqrt(HM_H / 1.25 * HM_W / 1.25)   # 44.3405 for 64x48 (oks_map.py)
TEMPERATURE = 0.5            # ProbMapHead default
SMOOTHING = 0.05             # config smoothing_weight


def _oks_s(k: int) -> float:
    """Per-keypoint sigma s (oks_map.py generate_probmaps): s = clip(
    (2σ_k)² * bbox_area * 2, 0.55, 3.0)."""
    s = (PT18_SIGMAS[k] * 2) ** 2 * BBOX_AREA * 2
    return float(np.clip(s, 0.55, 3.0))


# ----------------------------------------------------------------------------
def _jitter_box(box, strength: float = 1.0) -> list:
    """Imperfect-box noise (MIPNet-style): center shift up to ±10%·(w,h)·s,
    per-axis scale ±15%·s. Keypoints stay at GT image positions — only the
    crop window moves, so the model sees the person off-center / off-scale,
    like crops from real (imperfect) detections. MIPNet, ICCV 2021."""
    x, y, w, h = [float(v) for v in box[:4]]
    dx = (np.random.rand() * 2 - 1) * 0.1 * strength * w
    dy = (np.random.rand() * 2 - 1) * 0.1 * strength * h
    sw = 1.0 + (np.random.rand() * 2 - 1) * 0.15 * strength
    sh = 1.0 + (np.random.rand() * 2 - 1) * 0.15 * strength
    cx, cy = x + w * 0.5 + dx, y + h * 0.5 + dy
    w2, h2 = w * sw, h * sh
    return [cx - w2 * 0.5, cy - h2 * 0.5, w2, h2]


# Dataset: OKS-shaped heatmap targets + in_image flags (ProbMap.encode)
# ----------------------------------------------------------------------------
class MipNetDataset(tv.UPARDataset):
    """ProbPoseDataset + MIPNet-style box-noise augmentation: jitter the
    detection box itself (translation + per-axis scale), not just crop
    scale, so the network tolerates imperfect boxes at test time."""

    box_jitter_prob = 0.5
    box_jitter_strength = 1.0

    def __getitem__(self, idx):
        # copy of the parent flow (train_vitpose.py __getitem__) with
        # OKS-shaped targets instead of sigma-2 gaussians.
        a, im = self.items[idx]
        img = self._load_image(im["file_name"])
        kps = np.array(a["keypoints"], dtype=np.float32).reshape(-1, 3)
        bbox = a["bbox"]
        if self.augment and np.random.rand() < self.box_jitter_prob:
            bbox = _jitter_box(bbox, self.box_jitter_strength)

        center, scale = box_to_center_scale(bbox, self.out_w, self.out_h)

        if self.augment:
            sf = np.clip(np.random.randn() * 0.1 + 1.0, 1 - self.scale_factor, 1 + self.scale_factor)
            scale = scale * sf
            deg = (np.random.rand() * 2 - 1) * self.rotate_deg
            rad = math.radians(deg)
            cos, sin = math.cos(rad), math.sin(rad)
            rc = center
            rot = np.array([[cos, -sin, rc[0] - cos * rc[0] + sin * rc[1]],
                            [sin, cos, rc[1] - sin * rc[0] - cos * rc[1]]], dtype=np.float32)
            mat = tv.warp_matrix(center, scale, self.out_w, self.out_h)
            rot3 = np.vstack([rot, [0, 0, 1]])
            mat3 = np.vstack([mat, [0, 0, 1]])
            full = (mat3 @ rot3)
            crop = tv.warp_affine(img, full[:2].astype(np.float32), self.out_h, self.out_w)
            kps_t = tv.transform_kps(kps, full[:2].astype(np.float32))
            if np.random.rand() < self.flip_prob:
                crop = crop[:, ::-1]
                kps_t = kps_t.copy()
                kps_t[:, 0] = self.out_w - 1.0 - kps_t[:, 0]
                for a_, b_ in [(3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]:
                    kps_t[a_], kps_t[b_] = kps_t[b_].copy(), kps_t[a_].copy()
            if np.random.rand() < self.color_prob:
                alpha = 1.0 + (np.random.rand() * 0.4 - 0.2)
                beta = np.random.rand() * 0.2 - 0.1
                crop = crop * alpha + beta
        else:
            mat = tv.warp_matrix(center, scale, self.out_w, self.out_h)
            crop = tv.warp_affine(img, mat, self.out_h, self.out_w)
            kps_t = tv.transform_kps(kps, mat)

        crop = np.clip(crop, 0, 1)
        if self.augment and np.random.rand() < self.hemi_prob:
            side = np.random.choice(["l", "r"])
            nose_x = kps_t[0, 0] if kps_t[0, 2] > 0 else self.out_w / 2
            if side == "l":
                xs = np.arange(self.out_w) < nose_x - self.out_w * self.hemi_offset * 0
            else:
                xs = np.arange(self.out_w) > nose_x + self.out_w * self.hemi_offset * 0
            mask = np.tile(xs[None, :, None], (self.out_h, 1, 3)).astype(np.float32)
            crop = crop * (1 - mask)

        tensor = crop.transpose(2, 0, 1)
        tensor = (tensor - self.mean) / self.std

        # ---- OKS-shaped targets (generate_probmaps, heatmap coords) ----
        stride_y = (self.out_h - 1.0) / (self.heatmap_size[0] - 1.0)
        stride_x = (self.out_w - 1.0) / (self.heatmap_size[1] - 1.0)
        kps_hm = np.empty_like(kps_t[:, :2])
        kps_hm[:, 1] = kps_t[:, 1] / stride_y
        kps_hm[:, 0] = kps_t[:, 0] / stride_x

        hm = np.zeros((17, *self.heatmap_size), dtype=np.float32)
        yy, xx = np.mgrid[0:self.heatmap_size[0], 0:self.heatmap_size[1]]
        for i in range(17):
            if kps_t[i, 2] > 0:
                s = _oks_s(i)
                d2 = (xx - kps_hm[i, 0]) ** 2 + (yy - kps_hm[i, 1]) ** 2
                hm[i] = np.exp(-d2 / (2 * s))

        # annotated = labeled; in_image = crop coords within [0,W)x[0,H)
        annotated = (kps_t[:, 2] > 0).astype(np.float32)
        in_image = ((kps_t[:, 0] >= 0) & (kps_t[:, 0] < self.out_w) &
                    (kps_t[:, 1] >= 0) & (kps_t[:, 1] < self.out_h)).astype(np.float32)

        return {
            "pixel_values": torch.from_numpy(tensor.astype(np.float32)),
            "heatmaps": torch.from_numpy(hm),
            "annotated": torch.from_numpy(annotated),
            "in_image": torch.from_numpy(in_image),
            "kps_t": torch.from_numpy(kps_t.astype(np.float32)),
            "kps_hm": torch.from_numpy(kps_hm.astype(np.float32)),
            "image_id": im["id"],
            "anno_id": a["id"],
        }


# ----------------------------------------------------------------------------
# Sparsemax (probmap_head.py imports `from sparsemax import Sparsemax`;
# re-implemented here to avoid the extra dependency — standard algorithm,
# forward-only for training + decode)
# ----------------------------------------------------------------------------
class Sparsemax(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # sort descending along dim
        z = x.moveaxis(self.dim, -1)
        orig_shape = z.shape
        z_flat = z.reshape(-1, z.shape[-1])
        zs, _ = torch.sort(z_flat, descending=True, dim=-1)
        # k = [1..N] via ones+cumsum so device/dtype flow dynamically
        # (torch.arange(device=x.device) BAKES cpu into a traced graph —
        # CUDA eval then fails with device mismatch)
        k = torch.ones_like(zs[:1]).cumsum(dim=-1)
        zcum = torch.cumsum(zs, dim=-1)
        k_zs = k * zs
        # find support: zs > (zcum - 1) / k  (last index where strict)
        support = k_zs > zcum - 1.0
        # kmax = number of positive support elements per row
        kmax = support.to(zs.dtype).sum(dim=-1)
        # tau = (sum of top-kmax - 1) / kmax
        idx = (kmax - 1).long().clamp(min=0)
        tau = (torch.gather(zcum, 1, idx.unsqueeze(1)).squeeze(1) - 1.0) / kmax.clamp(min=1)
        out = torch.clamp(z_flat - tau.unsqueeze(1), min=0.0)
        return out.reshape(orig_shape).moveaxis(-1, self.dim)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class ProbPoseHead(nn.Module):
    """Sparsemax renormalization + in-image probability branch.

    Input: (B,17,64,48) heatmap features from the (finetuned) RemapHead.
    Output: heatmaps (B,17,64,48) — valid prob. mass, clamp(0,1);
            probs (B,17) — per-keypoint in-image probability (Sigmoid).
    The heatmap pathway is a 1x1 conv refine + Sparsemax(*normalize) +
    clamp, mirroring ProbMapHead.forward_heatmap semantics on top of an
    existing heatmap (their head deconv+conv+final then Sparsemax).
    """

    def __init__(self, num_joints=17, temperature=TEMPERATURE, normalize=1.0):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
        self.refine = nn.Conv2d(num_joints, num_joints, kernel_size=1)
        self.prob_pool = nn.Sequential(
            nn.Conv2d(num_joints, num_joints, 3, stride=1, padding=1),
            nn.BatchNorm2d(num_joints),
            nn.MaxPool2d(kernel_size=(4, 3), stride=(4, 3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_joints, num_joints, 3, stride=1, padding=1),
            nn.BatchNorm2d(num_joints),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_joints, num_joints, 3, stride=1, padding=1),
            nn.BatchNorm2d(num_joints),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_joints, num_joints, kernel_size=1),
        )
        # init: identity-ish for refine so we start near the finetuned model
        nn.init.dirac_(self.refine.weight)
        nn.init.zeros_(self.refine.bias)

    def forward(self, hm_in):
        x = self.refine(hm_in)
        b, c, h, w = x.shape
        xf = x.reshape(b, c, h * w)
        xf = self._sparsemax(xf / self.temperature)
        if self.normalize is not None:
            xf = xf * self.normalize
        heatmaps = torch.clamp(xf, 0, 1).reshape(b, c, h, w)

        # probability branch on DETACHED features (detach_probability=True)
        pd = self.prob_pool(hm_in.detach())
        probs = torch.sigmoid(pd).flatten(2).mean(dim=2)  # (B,17)
        return heatmaps, probs

    def _sparsemax(self, x):
        # same as Sparsemax module but on (B,C,N)
        zs, _ = torch.sort(x, descending=True, dim=-1)
        # ones_like+cumsum keeps device/dtype dynamic under torch.jit.trace
        k = torch.ones_like(zs[:1]).cumsum(dim=-1)
        zcum = torch.cumsum(zs, dim=-1)
        support = k * zs > zcum - 1.0
        kmax = support.to(zs.dtype).sum(dim=-1)
        idx = (kmax - 1).long().clamp(min=0)
        tau = (torch.gather(zcum, -1, idx.unsqueeze(-1)).squeeze(-1) - 1.0) / kmax.clamp(min=1)
        return torch.clamp(x - tau.unsqueeze(-1), min=0.0)


class ProbPoseViTPose(nn.Module):
    """ViTPose+ backbone + RemapHead + ProbPose head."""

    def __init__(self, vitpose, freeze_backbone=True):
        super().__init__()
        self.vitpose = vitpose
        self.remap = RemapHead()
        self.prob = ProbPoseHead()
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.vitpose.parameters():
                p.requires_grad = False

    def forward(self, pixel_values, dataset_index=None):
        if dataset_index is None:
            dataset_index = torch.zeros(pixel_values.shape[0], dtype=torch.int64,
                                        device=pixel_values.device)
        if self.freeze_backbone:
            with torch.no_grad():
                hm = self.vitpose(pixel_values=pixel_values,
                                  dataset_index=dataset_index).heatmaps  # (B,17,64,48)
        else:
            hm = self.vitpose(pixel_values=pixel_values,
                              dataset_index=dataset_index).heatmaps
        hm = self.remap(hm)
        # remap head already outputs 64x48 heatmaps — matches ProbPose's
        # config heatmap_size (48,64) after their patch16->deconv stack.
        heatmaps, probs = self.prob(hm)
        return heatmaps, probs


# ----------------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------------
class OKSHeatmapLoss(nn.Module):
    """ProbPose OKSHeatmapLoss, oks_type='minus', smoothing 0.05, per-pixel."""

    def __init__(self, smoothing_weight=SMOOTHING):
        super().__init__()
        self.smoothing_weight = smoothing_weight

    def forward(self, output, target, weights):
        # weights: (B,K) annotated mask
        B, K, H, W = output.shape
        oks_minus = output * (1 - target)
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                               dtype=torch.float32, device=output.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                               dtype=torch.float32, device=output.device).view(1, 1, 3, 3)
        gx = F.conv2d(output.reshape(B * K, 1, H, W), sobel_x, padding="same")
        gy = F.conv2d(output.reshape(B * K, 1, H, W), sobel_y, padding="same")
        gradient = (gx ** 2 + gy ** 2).reshape(B, K, H, W)

        w = (weights > 0).float()[:, :, None, None]
        denom = w.sum() * H * W  # per-pixel mean over annotated channels
        loss_map = (1 - self.smoothing_weight) * oks_minus + self.smoothing_weight * gradient
        return (loss_map * w).sum() / denom.clamp(min=1)


def bce_presence(pred_prob, target_in_image, annotated):
    w = (annotated > 0).float()
    denom = w.sum().clamp(min=1)
    loss = F.binary_cross_entropy(pred_prob, target_in_image.clamp(0, 1), reduction="none")
    return (loss * w).sum() / denom


# ----------------------------------------------------------------------------
# Quick val: px error via EV decode on val crops
# ----------------------------------------------------------------------------
_OKS_KERNELS = None


def _prepare_oks_kernels():
    global _OKS_KERNELS
    if _OKS_KERNELS is not None:
        return _OKS_KERNELS
    kernels = []
    for k in range(17):
        s = _oks_s(k)
        radius = int(math.ceil(s * 3))
        d = 2 * radius + 1
        c = d // 2
        dist_x = np.arange(d) - c
        dist_y = np.arange(d) - c
        dx, dy = np.meshgrid(dist_x, dist_y)
        dist = np.sqrt(dx ** 2 + dy ** 2)
        kern = np.exp(-(dist ** 2) / (2 * s))
        kern = (kern / kern.sum()).astype(np.float32)
        kernels.append(kern)
    _OKS_KERNELS = kernels
    return kernels


@torch.no_grad()
def _ev_decode_torch(hm: torch.Tensor) -> torch.Tensor:
    """Vectorized EV decode: per-kernel depthwise conv -> argmax -> Taylor
    subpixel (get_heatmap_expected_value). hm: (B,K,H,W) -> (B,K,2) hm coords."""
    B, K, H, W = hm.shape
    kernels = _prepare_oks_kernels()
    # pad each map by max kernel radius, conv with per-kernel canvas
    # (kernel centered in canvas), then crop back to H,W — equivalent to
    # scipy convolve(mode='reflect') but with replicate padding.
    max_r = max((kern.shape[0] - 1) // 2 for kern in kernels)
    padded = F.pad(hm, (max_r, max_r, max_r, max_r), mode="replicate")
    HP, WP = padded.shape[2] - 2 * max_r, padded.shape[3] - 2 * max_r  # conv out size = H,W
    convs = torch.zeros(B, K, HP, WP, device=hm.device, dtype=hm.dtype)
    for k in range(K):
        kern = torch.from_numpy(kernels[k]).to(hm.device, hm.dtype)
        r = (kern.shape[0] - 1) // 2
        canvas = torch.zeros((1, 1, 2 * max_r + 1, 2 * max_r + 1),
                             device=hm.device, dtype=hm.dtype)
        canvas[:, :, max_r - r:max_r + r + 1, max_r - r:max_r + r + 1] = kern.view(1, 1, *kern.shape)
        convs[:, k:k + 1] = F.conv2d(padded[:, k:k + 1], canvas, padding=0)
    flat = convs.reshape(B * K, H, W)
    # argmax + Taylor subpixel on flat
    idx = flat.reshape(B * K, H * W).argmax(dim=1)
    y0 = (idx // W).float()
    x0 = (idx % W).float()
    xi = idx // W
    xj = idx % W
    valid = (xj > 0) & (xj < W - 1) & (xi > 0) & (xi < H - 1)
    xi_s = xi.clamp(1, H - 2)
    xj_s = xj.clamp(1, W - 2)
    dx = (flat[torch.arange(B * K), xi_s, xj_s + 1] - flat[torch.arange(B * K), xi_s, xj_s - 1]) / 2.0
    dy = (flat[torch.arange(B * K), xi_s + 1, xj_s] - flat[torch.arange(B * K), xi_s - 1, xj_s]) / 2.0
    dxx = (flat[torch.arange(B * K), xi_s, xj_s + 1] + flat[torch.arange(B * K), xi_s, xj_s - 1]
           - 2 * flat[torch.arange(B * K), xi_s, xj_s])
    dyy = (flat[torch.arange(B * K), xi_s + 1, xj_s] + flat[torch.arange(B * K), xi_s - 1, xj_s]
           - 2 * flat[torch.arange(B * K), xi_s, xj_s])
    dxx = torch.where(dxx != 0, dxx, torch.full_like(dxx, 1e-6))
    dyy = torch.where(dyy != 0, dyy, torch.full_like(dyy, 1e-6))
    sx = torch.where(valid, -dx / dxx, torch.zeros_like(dx))
    sy = torch.where(valid, -dy / dyy, torch.zeros_like(dy))
    out = torch.stack([x0 + sx, y0 + sy], dim=-1).reshape(B, K, 2)
    return out


@torch.no_grad()
def quick_val(model, loader, device):
    model.eval()
    errs = []
    stride_y = (OUT_H - 1.0) / (HM_H - 1.0)
    stride_x = (OUT_W - 1.0) / (HM_W - 1.0)
    for batch in loader:
        px = batch["pixel_values"].to(device)
        kps_t = batch["kps_t"].to(device)
        ann = batch["annotated"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            hm_pred, _probs = model(px)
        locs = _ev_decode_torch(hm_pred.float())  # (B,K,2) heatmap coords
        pred_x = locs[..., 0] * stride_x
        pred_y = locs[..., 1] * stride_y
        for b in range(px.shape[0]):
            for k in range(17):
                if ann[b, k] > 0:
                    errs.append(math.hypot(float(pred_x[b, k] - kps_t[b, k, 0]),
                                           float(pred_y[b, k] - kps_t[b, k, 1])))
    return float(np.mean(errs)) if errs else 0.0


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--unfreeze-at-epoch", type=int, default=4)
    ap.add_argument("--out-dir", type=str, default="training/runs/final_pose")
    ap.add_argument("--box-jitter-prob", type=float, default=0.25)
    ap.add_argument("--box-jitter-strength", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--presence-weight", type=float, default=0.1)
    ap.add_argument("--base-model", default="usyd-community/vitpose-plus-small")
    ap.add_argument("--base-revision", default="0c30b6534bb621af0162b481176742577264e36e")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    print("[*] loading annotations ...", flush=True)
    val_split = json.load(open(os.path.join(os.path.dirname(__file__), "..", "eval", "val_split.json")))
    val_ids = set(sum(val_split.values(), []))
    MipNetDataset.box_jitter_prob = args.box_jitter_prob
    MipNetDataset.box_jitter_strength = args.box_jitter_strength
    train_ds = MipNetDataset(ANN_FILE, DATA_ROOT, augment=True, val_ids=val_ids, is_val=False)
    val_ds = MipNetDataset(ANN_FILE, DATA_ROOT, augment=False, val_ids=val_ids, is_val=True)
    print(f"[*] train={len(train_ds)} val={len(val_ds)}", flush=True)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True, drop_last=True,
                                               persistent_workers=True, prefetch_factor=4)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)

    print("[*] loading ViTPose+ small ...", flush=True)
    from transformers import VitPoseForPoseEstimation
    vitpose = VitPoseForPoseEstimation.from_pretrained(args.base_model, revision=args.base_revision if not os.path.isdir(args.base_model) else None).to(device)
    model = ProbPoseViTPose(vitpose, freeze_backbone=True).to(device)

    prob_params = [p for n, p in model.prob.named_parameters()]
    remap_params = list(model.remap.parameters())
    opt = torch.optim.AdamW([
        {"params": prob_params, "lr": args.lr},
        {"params": remap_params, "lr": args.lr * 0.5},
    ], weight_decay=1e-4)

    oks_loss = OKSHeatmapLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    best_metric = float("inf")
    history = []
    for epoch in range(args.epochs):
        if epoch == args.unfreeze_at_epoch:
            print(f"[*] unfreezing backbone at epoch {epoch}", flush=True)
            model.freeze_backbone = False
            for p in model.vitpose.parameters():
                p.requires_grad = True
            opt = torch.optim.AdamW([
                {"params": model.prob.parameters(), "lr": args.lr * 0.1},
                {"params": model.remap.parameters(), "lr": args.lr * 0.05},
                {"params": model.vitpose.parameters(), "lr": 1e-5},
            ], weight_decay=1e-4)
            scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

        model.train()
        t0 = time.time(); run_loss = 0.0; nb = 0
        for it, batch in enumerate(train_loader):
            px = batch["pixel_values"].to(device, non_blocking=True)
            hm = batch["heatmaps"].to(device, non_blocking=True)
            ann = batch["annotated"].to(device, non_blocking=True)
            in_img = batch["in_image"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
                hm_pred, probs = model(px)
                loss_oks = oks_loss(hm_pred.float(), hm.float(), ann.float())
            # BCE is autocast-unsafe: compute presence loss in fp32 outside
            # the autocast region (F.binary_cross_entropy raises under autocast).
            loss_pres = bce_presence(probs.float(), in_img.float(), ann.float())
            loss = loss_oks + args.presence_weight * loss_pres
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            run_loss += loss.item(); nb += 1
            if it % 50 == 0:
                print(f"  ep{epoch} it{it}/{len(train_loader)} oks={loss_oks.item():.4f} "
                      f"pres={loss_pres.item():.4f} ({(time.time()-t0)/(it+1):.2f}s/it)", flush=True)
        train_loss = run_loss / max(nb, 1)

        val_px_err = quick_val(model, val_loader, device)
        print(f"[*] epoch {epoch}: train_loss={train_loss:.5f} val_px_err={val_px_err:.3f} ({time.time()-t0:.0f}s)", flush=True)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_px_err": val_px_err})
        torch.save(model.state_dict(), os.path.join(args.out_dir, "last.pt"))
        if val_px_err < best_metric:
            best_metric = val_px_err
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pt"))
            print(f"[*] new best (val_px_err={val_px_err:.3f}) saved", flush=True)
        with open(os.path.join(args.out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()