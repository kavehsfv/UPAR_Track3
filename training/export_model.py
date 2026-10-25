"""Export the ProbPose-fine-tuned ViTPose to TorchScript for Codabench.

Traces a wrapper whose forward returns ONLY the (heatmaps, probs) pair —
the ProbPose head output. The submission decodes coordinates via the EV
decode (OKS-kernel convolution + argmax + Taylor subpixel) implemented in
run.py, and uses the presence probs for the v bit.

NOTE: torch.jit.trace cannot trace the sparsemax SORT (dynamic shapes are
fine but the sort/cumsum/gather chain in trace mode is supported — verify
via reload diff). Must run with CUDA_VISIBLE_DEVICES="" to trace on CPU.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "train_pose", os.path.join(_HERE, "train_pose.py"))
tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tp)

ProbPoseViTPose = tp.ProbPoseViTPose


class ProbWrapper(torch.nn.Module):
    """Trace-friendly wrapper: registered dataset_index buffer, tuple output."""

    def __init__(self, model: ProbPoseViTPose):
        super().__init__()
        self.model = model
        self.register_buffer("dataset_index", torch.zeros(1, dtype=torch.int64))

    def forward(self, pixel_values):
        dataset_index = self.dataset_index.expand(pixel_values.shape[0])
        heatmaps, probs = self.model(pixel_values, dataset_index=dataset_index)
        return heatmaps, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out-dir", type=str,
                    default=os.path.join(_HERE, "..", "assets"))
    args = ap.parse_args()

    print("[*] loading ViTPose+ small base ...", flush=True)
    from transformers import VitPoseConfig, VitPoseForPoseEstimation
    config = VitPoseConfig.from_pretrained(os.path.join(_HERE, "model_config"), local_files_only=True)
    vitpose = VitPoseForPoseEstimation(config)
    os.makedirs(args.out_dir, exist_ok=True)
    model = ProbPoseViTPose(vitpose, freeze_backbone=True)

    state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    state = {k: v for k, v in state.items() if not k.startswith("freeze_backbone")}
    missing, unexpected = model.load_state_dict(state, strict=True)
    print(f"[*] loaded ckpt: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("    missing keys (first 5):", missing[:5])
    if unexpected:
        print("    unexpected keys (first 5):", unexpected[:5])
    model.eval()

    wrapper = ProbWrapper(model).eval()

    example = torch.zeros(1, 3, 256, 192)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example)
        hm, pr = traced(example)
        b4 = traced(torch.zeros(4, 3, 256, 192))
    print(f"[*] sanity: heatmaps {tuple(hm.shape)} probs {tuple(pr.shape)}; batch4 ok={tuple(b4[0].shape)}")
    print(f"    heatmap sum/channel (should be 1.0): {hm.sum((2,3))[0,:3]}")
    print(f"    probs range: [{pr.min():.4f}, {pr.max():.4f}]")

    out_path = os.path.join(args.out_dir, "mipnet_ft.torchscript")
    traced.save(out_path)
    print(f"[*] saved {out_path} ({os.path.getsize(out_path)/1e6:.0f} MB)")

    meta = {
        "input_width": 192,
        "input_height": 256,
        "normalize_factor": 200.0,
        "padding_factor": 1.25,
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
        "hm_height": 64,
        "hm_width": 48,
        "temperature": 0.5,
        "normalize": 1.0,
        # PT18 sigmas for the EV-decode kernels (run.py rebuilds kernels)
        "pt18_sigmas": [float(s) for s in tp.PT18_SIGMAS],
        "bbox_area": float(tp.BBOX_AREA),
    }
    meta_path = os.path.join(args.out_dir, "preprocess_mip.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[*] wrote {meta_path}")

    # reload + cross-check on CUDA if available
    reloaded = torch.jit.load(out_path, map_location="cpu")
    with torch.no_grad():
        r = reloaded(example)
    diff_hm = float((r[0] - hm).abs().max())
    diff_pr = float((r[1] - pr).abs().max())
    print(f"[*] reload check: max|diff_hm| = {diff_hm:.2e}, max|diff_pr| = {diff_pr:.2e}")
    if torch.cuda.is_available():
        reloaded_cuda = torch.jit.load(out_path, map_location="cuda")
        with torch.no_grad():
            rc = reloaded_cuda(example.cuda())
        print(f"[*] cuda check: {tuple(rc[0].shape)} on {rc[0].device}")


if __name__ == "__main__":
    main()