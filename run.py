"""UPAR Challenge 2027 Track 3 submission — v3: mipnet_v2 base + best-of-series
eval-side TTA (multi-scale + flip) and expected-OKS scoring.

Training-side v3 upgrades (training/train_v3.py): EMA weight averaging
(RTMPose), cosine LR decay (RTMPose), 2-stage aug narrowing (RTMPose) on the
mipnet_v2 quarter-dose box-jitter recipe (OKS targets + Sparsemax + EV decode).

Inference protocol (ProbPose, CVPR 2025):
- crop 256x192 (same box->center/scale as training)
- model outputs (heatmaps, probs): sparsemax-normalized OKS heatmaps
  (B,17,64,48) with valid probability mass per channel, and per-keypoint
  in-image presence probabilities (B,17)
- FLIP TEST as ProbMapHead.predict (flip_mode='heatmap', shift_heatmap=
  False): average heatmaps of orig + flipped crop (channels swapped via
  FLIP_INDICES, x-axis reversed); presence probs averaged likewise
- decode: get_heatmap_expected_value — convolve each channel with its
  normalized OKS kernel (s = clip((2σ_k)²·bbox_area·2, 0.55, 3.0)),
  argmax of the CONVOLVED map, then Taylor sub-pixel refinement
- v bit: 1.0 if presence prob >= 0.5 else 0.0 (protocol wants v in {0,1,2};
  in-image presence is the only signal we have)
- score: mean heatmap value at the decoded locations (ordering-only)
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "assets", "mipnet_ft.torchscript")
META_PATH = os.path.join(HERE, "assets", "preprocess_mip.json")
PRIOR_PATH = os.path.join(HERE, "assets", "mean_pose.json")

BATCH_SIZE = 8
TTA_SCALES = (0.9, 1.0, 1.1)   # multi-scale crops averaged in image space
FLIP_SHIFT = False             # falsified: +1px shift hurts EV decode (-7 AP95)

# full 17-slot flip mapping: slot i -> slot of its symmetric partner
# (ears 3<->4 swap; nose/head slots are self-symmetric)
FLIP_INDICES = [0, 1, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

_MODEL = None
_DEVICE = None
_META: dict[str, Any] = {}
_PRIOR: list[list[float]] | None = None
_MEAN: np.ndarray | None = None
_STD: np.ndarray | None = None
_OKS_KERNELS: list[np.ndarray] | None = None


def _oks_s(k: int) -> float:
    sigmas = _META["pt18_sigmas"]
    s = (sigmas[k] * 2) ** 2 * float(_META["bbox_area"]) * 2
    return float(np.clip(s, 0.55, 3.0))


def _prepare_oks_kernels() -> list[np.ndarray]:
    """Normalized OKS kernels per keypoint (post_processing.py
    _prepare_oks_kernels): kernel = exp(-d²/2s) / sum."""
    global _OKS_KERNELS
    if _OKS_KERNELS is not None:
        return _OKS_KERNELS
    import math
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
        kernels.append((kern / kern.sum()).astype(np.float32))
    _OKS_KERNELS = kernels
    return kernels


def _convolve_reflect(hm: np.ndarray, kern: np.ndarray) -> np.ndarray:
    """scipy.ndimage.convolve(mode='reflect') equivalent: np.pad 'reflect'
    + 'valid' correlation. ProbPose kernels are symmetric, so correlation ==
    convolution."""
    r = (kern.shape[0] - 1) // 2
    padded = np.pad(hm, r, mode="reflect")
    from numpy.lib.stride_tricks import sliding_window_view
    win = sliding_window_view(padded, (kern.shape[0], kern.shape[1]))
    return (win * kern[None, None]).sum(axis=(2, 3)).astype(np.float32)


def load_model() -> None:
    global _MODEL, _DEVICE, _META, _PRIOR, _MEAN, _STD
    if _MODEL is not None:
        return

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"{MODEL_PATH} is missing. Run `python export_probpose.py` before zipping: "
            "the evaluation container cannot download anything."
        )

    with open(META_PATH, "r", encoding="utf-8") as fh:
        _META = json.load(fh)
    with open(PRIOR_PATH, "r", encoding="utf-8") as fh:
        _PRIOR = json.load(fh)["mean_pose_normalized"]

    _MEAN = np.asarray(_META["image_mean"], dtype=np.float32).reshape(3, 1, 1)
    _STD = np.asarray(_META["image_std"], dtype=np.float32).reshape(3, 1, 1)

    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    _MODEL = torch.jit.load(MODEL_PATH, map_location=_DEVICE)
    _MODEL.eval()
    print(f"[submission] ProbPose fine-tuned ViTPose ready on {_DEVICE} "
          f"({_META['input_width']}x{_META['input_height']} input)",
          flush=True)


def box_to_center_scale(box, out_w: int, out_h: int,
                        normalize_factor: float, padding_factor: float):
    x, y, w, h = [float(v) for v in box[:4]]
    aspect = out_w / out_h
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
    if w > aspect * h:
        h = w / aspect
    elif w < aspect * h:
        w = h * aspect
    scale = np.array([w / normalize_factor, h / normalize_factor], dtype=np.float32)
    return center, scale * padding_factor


def warp_matrix(center: np.ndarray, scale: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
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


def warp_affine(image: np.ndarray, matrix: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    bottom = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    inverse = np.linalg.inv(np.vstack([matrix, bottom]))[:2].astype(np.float32)
    ys, xs = np.meshgrid(np.arange(out_h, dtype=np.float32),
                         np.arange(out_w, dtype=np.float32), indexing="ij")
    src_x = inverse[0, 0] * xs + inverse[0, 1] * ys + inverse[0, 2]
    src_y = inverse[1, 0] * xs + inverse[1, 1] * ys + inverse[1, 2]

    height, width = image.shape[:2]
    x0 = np.floor(src_x).astype(np.int64)
    y0 = np.floor(src_y).astype(np.int64)
    x1, y1 = x0 + 1, y0 + 1
    wx, wy = src_x - x0, src_y - y0

    inside = (src_x >= 0) & (src_x <= width - 1) & (src_y >= 0) & (src_y <= height - 1)
    x0c, x1c = np.clip(x0, 0, width - 1), np.clip(x1, 0, width - 1)
    y0c, y1c = np.clip(y0, 0, height - 1), np.clip(y1, 0, height - 1)

    top = image[y0c, x0c] * (1 - wx)[..., None] + image[y0c, x1c] * wx[..., None]
    bottom = image[y1c, x0c] * (1 - wx)[..., None] + image[y1c, x1c] * wx[..., None]
    return (top * (1 - wy)[..., None] + bottom * wy[..., None]) * inside[..., None]


def preprocess(image: np.ndarray, box) -> tuple:
    """image: float32 RGB in [0,1] (fine-tune normalization; no 255 rescale)."""
    out_w, out_h = int(_META["input_width"]), int(_META["input_height"])
    center, scale = box_to_center_scale(
        box, out_w, out_h, float(_META["normalize_factor"]), float(_META["padding_factor"]))
    crop = warp_affine(image, warp_matrix(center, scale, out_w, out_h), out_h, out_w)
    tensor = crop.transpose(2, 0, 1)
    return (((tensor - _MEAN) / _STD).astype(np.float32), center, scale)


def _flip_test_heatmaps(batch: torch.Tensor):
    """ProbMapHead.predict flip TTA (flip_mode='heatmap', shift_heatmap=False):
    average heatmaps of orig + flipped crop; channels swapped by FLIP_INDICES,
    x-axis reversed. Presence probs averaged the same way.

    PERFORMANCE: orig and flipped crops are stacked into ONE model forward
    (batch 2B) instead of two forwards — halves forward count; combined with
    the multi-scale stacking in predict_image the whole TTA is 1 forward."""
    B = batch.shape[0]
    both = torch.cat([batch, torch.flip(batch, dims=[3])], dim=0)
    hm_all, pr_all = _MODEL(both)
    hm, pr = hm_all[:B], pr_all[:B]
    hmf, prf = hm_all[B:], pr_all[B:]

    hmf_np = hmf.float().cpu().numpy()          # (B,17,64,48)
    hmf_np = hmf_np[:, FLIP_INDICES, :][:, :, :, ::-1]
    if FLIP_SHIFT:
        # UDP unbiased flip: the flipped crop puts content ~1px off; shift the
        # flipped map right by 1 px to realign (shift_heatmap in UDP-Pose).
        hmf_np[:, :, :, 1:] = hmf_np[:, :, :, :-1]
    hmf = torch.from_numpy(hmf_np.copy()).to(hm.device)

    prf_np = prf.float().cpu().numpy()[:, FLIP_INDICES]
    prf = torch.from_numpy(prf_np.copy()).to(pr.device)

    avg_hm = (hm + hmf) * 0.5
    avg_pr = (pr + prf) * 0.5
    return avg_hm, avg_pr


def decode_ev(hm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """get_heatmap_expected_value: per-channel OKS-kernel convolve -> argmax
    of convolved map -> Taylor subpixel. hm (B,17,64,48) -> locs (B,17,2)
    [x,y] in heatmap coords, vals (B,17) = CONVOLVED peak = expected OKS."""
    B, K, H, W = hm.shape
    kernels = _prepare_oks_kernels()
    locs = np.zeros((B, K, 2), dtype=np.float32)
    vals = np.zeros((B, K), dtype=np.float32)
    for b in range(B):
        for k in range(K):
            conv = _convolve_reflect(hm[b, k], kernels[k])
            idx = int(np.argmax(conv))
            yi, xi = divmod(idx, W)
            # Taylor subpixel on the CONVOLVED map
            if 0 < xi < W - 1 and 0 < yi < H - 1:
                dx = (conv[yi, xi + 1] - conv[yi, xi - 1]) / 2.0
                dy = (conv[yi + 1, xi] - conv[yi - 1, xi]) / 2.0
                dxx = conv[yi, xi + 1] + conv[yi, xi - 1] - 2 * conv[yi, xi]
                dyy = conv[yi + 1, xi] + conv[yi - 1, xi] - 2 * conv[yi, xi]
                if dxx != 0:
                    xi -= dx / dxx
                if dyy != 0:
                    yi -= dy / dyy
            locs[b, k, 0] = xi
            locs[b, k, 1] = yi
            vals[b, k] = conv[int(np.clip(round(yi), 0, H - 1)),
                              int(np.clip(round(xi), 0, W - 1))]
    return locs, vals


def _prior_pose(box) -> list[list[float]]:
    bx, by, bw, bh = [float(v) for v in box]
    return [[bx + u * bw, by + v * bh, 1.0] for u, v in _PRIOR]


def _resolve_image_path(sample: dict[str, Any]) -> str:
    image_path = str(sample.get("image_path", ""))
    if image_path and os.path.isfile(image_path):
        return image_path

    file_name = str(sample.get("file_name", ""))
    if image_path:
        candidates = [
            image_path.replace("/images/images/", "/images/"),
            image_path.replace("\\images\\images\\", "\\images\\"),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate

    if file_name:
        candidates = [
            file_name,
            file_name.removeprefix("images/"),
            file_name.removeprefix("images\\"),
        ]
        for candidate in candidates:
            candidate_path = os.path.join(HERE, candidate)
            if os.path.isfile(candidate_path):
                return candidate_path

    if image_path:
        return image_path
    raise FileNotFoundError("Could not resolve image path for sample.")


def predict_image(sample: dict[str, Any]) -> dict[str, Any]:
    if _MODEL is None:
        load_model()

    boxes = sample["boxes"]
    if not boxes:
        return {"keypoints": [], "scores": []}

    image = np.asarray(Image.open(_resolve_image_path(sample)).convert("RGB"), dtype=np.float32) / 255.0

    keypoints: list[Any] = [None] * len(boxes)
    scores: list[float] = [0.0] * len(boxes)

    usable = [i for i, b in enumerate(boxes) if float(b[2]) > 0 and float(b[3]) > 0]
    for i in set(range(len(boxes))) - set(usable):
        keypoints[i] = _prior_pose(boxes[i])
        scores[i] = 0.01

    out_w, out_h = int(_META["input_width"]), int(_META["input_height"])
    hm_h, hm_w = int(_META["hm_height"]), int(_META["hm_width"])

    for start in range(0, len(usable), BATCH_SIZE):
        chunk = usable[start:start + BATCH_SIZE]
        tensors, centers, scales = [], [], []
        for i in chunk:
            tensor, center, scale = preprocess(image, boxes[i])
            tensors.append(tensor)
            centers.append(center)
            scales.append(scale)

        # ---------------- multi-scale TTA (decode-then-average) ----------------
        # For each scale factor, crop with a rescaled padded box, flip-test,
        # EV-decode, and map to image px; final keypoint = MEAN over scales.
        # Score uses the CONVOLVED peak (expected OKS at the decoded loc).
        # Build ALL crops first (len(chunk) boxes x len(TTA_SCALES) scales),
        # then run the flip TTA on the whole stacked batch — ONE model call
        # for the entire multi-scale TTA (6 forwards -> 1).
        all_tensors, all_centers, all_scales = [], [], []
        for sf in TTA_SCALES:
            for i in chunk:
                box_s = list(boxes[i])
                bx, by, bw, bh = [float(v) for v in box_s[:4]]
                cx, cy = bx + bw * 0.5, by + bh * 0.5
                bw2, bh2 = bw * sf, bh * sf
                box_s = [cx - bw2 * 0.5, cy - bh2 * 0.5, bw2, bh2] + list(box_s[4:])
                tensor, center, scale = preprocess(image, box_s)
                all_tensors.append(tensor)
                all_centers.append(center)
                all_scales.append(scale)

        batch = torch.from_numpy(np.stack(all_tensors)).to(_DEVICE)
        avg_hm, avg_pr = _flip_test_heatmaps(batch)  # internally 1 forward

        hm_np = avg_hm.float().cpu().numpy()              # (nbox*nscale, 17, 64, 48)
        pr_np_all = avg_pr.float().cpu().numpy()
        locs, vals = decode_ev(hm_np)

        # heatmap coords -> crop px -> image px, then average over scales
        ns = len(TTA_SCALES)
        nb = len(chunk)
        coords_all = np.zeros_like(locs)
        for j in range(coords_all.shape[0]):
            sc = all_scales[j] * 200.0
            crop_x = locs[j, :, 0] * (out_w - 1.0) / (hm_w - 1.0)
            crop_y = locs[j, :, 1] * (out_h - 1.0) / (hm_h - 1.0)
            coords_all[j, :, 0] = crop_x * sc[0] / (out_w - 1.0) + all_centers[j][0] - sc[0] * 0.5
            coords_all[j, :, 1] = crop_y * sc[1] / (out_h - 1.0) + all_centers[j][1] - sc[1] * 0.5
        # layout is scale-major: [s0b0..s0bN, s1b0..s1bN, s2b0..s2bN]
        coords = coords_all.reshape(ns, nb, 17, 2).mean(axis=0)
        vals = vals.reshape(ns, nb, 17).mean(axis=0)
        pr_np = pr_np_all.reshape(ns, nb, 17).mean(axis=0)

        for j, i in enumerate(chunk):
            # clamp to image bounds (challenge wants absolute in-image px)
            h_img, w_img = image.shape[:2]
            clamped = np.clip(coords[j], [0.0, 0.0], [w_img - 1.0, h_img - 1.0])
            # v bit from presence probability (in-image)
            vs = [1.0 if p >= 0.5 else 0.0 for p in pr_np[j]]
            keypoints[i] = [[float(x), float(y), v] for (x, y), v in zip(clamped, vs)]
            scores[i] = float(np.mean(vals[j]))

    return {"keypoints": keypoints, "scores": scores}
