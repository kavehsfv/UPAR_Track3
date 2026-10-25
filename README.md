# UPAR Challenge 2027 — Track 3 release

**Participant:** Kaveh Safavigerdini, Electrical Engineering and Computer Science Department,
University of Missouri–Columbia, USA. Contact: ksgh2@umsystem.edu.
**Codabench:** `kavehsfv-994308-994308_VID-Docker-GPU`.

This repository releases the latest locally finalized **mipnet_v3** inference package,
its **mipnet_v2 training recipe**, trained state dictionary and deployment weights.
It contains only this method, not the experimental paper-tree series.

## 1. Official result and submission identity

| Metric, test set (36,509) | Organizer leaderboard export |
|---|---:|
| AP (mean) | **0.8862830000** |
| AP50 | 0.9672830000 |
| AP_medium | 0.8564020000 |
| AP_large | 0.6790190000 |

Source: participant-supplied organizer CSV, preserved in [evidence/leaderboard.csv](evidence/leaderboard.csv).
The participant confirms that the best/final method was uploaded to the challenge.
The table above is the official leaderboard result for that final entry. This release
provides its runtime-fixed `task3_mipnet_v3.zip` inference implementation (original
archive MD5 **2de661186c5954f38c8f2bd6cdd4fb96**), training code and weights.
Local compatibility, checkpoint export and inference-payload checks are recorded
in `evidence/`; no new hidden-server evaluation was performed during packaging.

## 2. Download code AND trained weights (no inference downloads)

```bash
# Install Git LFS first: https://git-lfs.com/
git lfs install
git clone https://github.com/kavehsfv/UPAR_Track3.git
cd UPAR_Track3
git lfs pull
python3 -c "from pathlib import Path; assert Path('assets/mipnet_ft.torchscript').stat().st_size > 100_000_000"
```

The two Git LFS files are `assets/mipnet_ft.torchscript` (133,223,636 bytes,
ready-to-run TorchScript weights) and `weights/pose_model.pt` (132,745,507 bytes,
final training state dictionary, selected at zero-based epoch 13 of 15).
Small preprocessing and fallback-pose JSONs are ordinary git files. SHA256
checksums are in [evidence/artifacts.json](evidence/artifacts.json).
GitHub's source ZIP may contain LFS pointers; use `git lfs pull` or the downloadable
`Track3_weights.zip` attached at https://github.com/kavehsfv/UPAR_Track3/releases/tag/v1.0 .
Unzip that bundle at the repository root to populate `assets/` and `weights/`.

## 3. Inference and packaging

Requires Python 3.11, PyTorch, NumPy and Pillow only. **No transformers, mmpose,
OpenCV, scipy or network access at inference.** CPU and CUDA are automatically selected.

```bash
# Challenge-compatible CPU test environment (use a virtual environment):
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.4.6 Pillow==12.3.0
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python validate_release.py
python build_submission.py
python validate_release.py --zip submission_zip/task3_mipnet_v3.zip
```

The rebuilt ZIP has the **same inference payload bytes** as the original archive.
Its ZIP-container checksum differs because timestamps are normalized and the stale,
unused `export_v3.py` experiment helper is omitted. Training/export code instead
lives in `training/` and is not used by Codabench.

Server interface: `run.load_model()`, then `run.predict_image(sample)`, where sample
contains image_path, id, file_name, width, height, boxes (xywh) and box_ids.
Return value: `{"keypoints": [[[x,y,v], ...17 joints...], ...], "scores": [...]}`.
All coordinates are absolute image pixels. Ear slots remain in the 17-slot output.
No detector is used: boxes are supplied by the challenge.

```bash
# Reproducible inference container; download dependencies at BUILD time only.
docker build -t upar-track3 -f Dockerfile .
docker run --rm --network none upar-track3
# Optional GPU (requires NVIDIA container toolkit):
docker run --rm --gpus all --network none upar-track3
```

Docker is not installed on the release-preparation host, so an actual Docker build/run
has **not** been tested. The identical Python inference payload is tested separately
under PyTorch 2.4.1 CPU; see `evidence/validation_cpu.txt`.

## 4. Submitted-method implementation

A ViTPose+ Small backbone with COCO expert route 0 feeds its pretrained heatmap
decoder, a PT18 remapping/refinement head, a 1x1 refinement and Sparsemax
(temperature 0.5, spatial mass 1). A detached feature branch predicts in-crop
presence probabilities. Input is RGB 256x192, aspect-corrected box padding 1.25,
bilinear affine resampling and ImageNet normalization.

Training uses OKS-shaped targets and an expected-OKS loss with Sobel smoothing
(weight 0.05), plus presence BCE weighted 0.1. Box jitter occurs with p=0.25:
center shifts are uniform within ±5% of box width/height and per-axis sizes within
±7.5%. Other augmentation: horizontal flip 0.5, clipped Gaussian scale
(N(1,0.1), [0.7,1.3]), rotation ±30 degrees, brightness/contrast 0.5,
and one-sided occlusion 0.3. This is a **box-noise adaptation**, not the original
MIPNet multi-instance network.

Inference uses crop scales **0.9, 1.0, 1.1**, each with original+flip testing.
All six views are stacked into a single model call per box chunk. Flipped
heatmaps are mirrored and left/right channels swapped **without a one-pixel shift**.
Each scale's averaged heatmap is convolved with its normalized OKS kernel;
argmax plus Taylor subpixel refinement is mapped back to image space. Coordinates,
convolved-peak confidence and presence probabilities are averaged across scales.
The per-box score averages the 17 convolved peaks; v=1 when presence >=0.5,
otherwise 0. This score is not a calibrated error probability. Chunk size is 8
boxes (up to 48 view crops). Degenerate boxes use the included training-pose prior.

**Important:** `training/train_pose.py` is the actual mipnet_v2 recipe. The unused
v3 EMA/cosine/narrow-augmentation experiment is deliberately not included.
The byte-preserved `run.py` header still mentions those rejected experiments;
its executable configuration and the supplied weights implement the method above.
The training recipe is defined by `training/train_pose.py`, not that legacy header.

## 5. Training reproduction

Use the organizer's licensed Track 3 development data from the official starter
repository: https://github.com/MickaelCormier/UPAR-Challenge-2027-Track3 .
Images and annotations are not redistributed here. Preserve this layout:

```text
/path/to/upar-data/
  annotations/specialization/train_upar_all.json
  market/bounding_box_train/... (and other referenced Market subfolders)
  pa100k/images/...
  peta/images/... (if referenced by your organizer release)
```

The actual annotation snapshot used contains 42,736 annotations/42,826 image
records; 301 annotation IDs in `eval/val_split.json` are held out, leaving **42,435**
training annotations (29,613 PA-100K, 12,822 Market). No PETA annotations occur in
this particular combined training snapshot. No official validation or hidden test
images are used for gradient updates. The local holdout is used for checkpoint
selection, not training. Data paths are portable via `UPAR_DATA`.

**Historical training stack:** Python 3.11, torch 2.14.0+cu130,
transformers 5.18.0, NumPy 2.3.5, Pillow 12.3.0. PyTorch 2.4.1 is the
*inference* compatibility target, not the transformers training environment.

```bash
# In a separate training environment with suitable CUDA 13 driver/GPU:
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-training.txt
UPAR_DATA=/path/to/upar-data python training/train_pose.py \
  --epochs 15 --batch-size 64 --lr 1e-3 --unfreeze-at-epoch 4 \
  --box-jitter-prob 0.25 --box-jitter-strength 0.5 --seed 0 --workers 8 \
  --out-dir training/runs/final_pose
# Requires initial network/cache access for the pinned starter checkpoint only.
# --base-model /path/to/local/checkpoint enables an offline warm start.

# Export with CPU tracing; local config + full state means NO download:
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python training/export_model.py --ckpt training/runs/final_pose/best.pt \
  --out-dir assets
# Or use --ckpt weights/pose_model.pt to re-export supplied final weights.
```

Optimizer: AdamW weight decay 1e-4. Epochs 0–3: pretrained network frozen,
probability/refinement branch LR 1e-3 and remap LR 5e-4. At epoch 4 unfreeze:
branch LR 1e-4, remap LR 5e-5, pretrained network LR 1e-5. No cosine scheduler
or EMA in the selected training recipe. BF16 autocast, FP32 presence BCE,
batch 64, drop_last=True, seed 0. Checkpoint selection minimizes held-out
crop-space pixel error (best 3.452812 at epoch 13). Full historical training
curve: `evidence/training_history.json`. Floating-point and dataloader variation
can prevent bitwise reproduction by retraining; the supplied checkpoint is the
reference for exact inference.

```bash
# Optional training container (not executed on the preparation host):
docker build -t upar-track3-train -f Dockerfile.train .
docker run --rm --gpus all --ipc=host \
  -v /path/to/upar-data:/data:ro -v "$PWD/training/runs:/workspace/training/runs" \
  upar-track3-train
```

Historical paper-tree local reports used a non-COCO OKS/AP reduction. They are
kept only as diagnostic provenance (`evidence/historical_local_diagnostic.json`),
**not** as official AP claims. The server's hidden data and exact scorer were
not available for re-evaluation during release preparation.

## 6. Pretraining disclosure / organizer clarification needed

The checkpoint is the same public **usyd-community/vitpose-plus-small** used by
the organizer starter, pinned to revision **0c30b6534bb621af0162b481176742577264e36e**.
Routing expert 0 at inference is **not** proof of COCO-only pretraining: shared
weights can retain supervision from other datasets. The public model card
(https://huggingface.co/usyd-community/vitpose-plus-small) describes multi-dataset
training, including COCO, AI Challenger and MPII, and the checkpoint config has
six experts. The starter README separately says only COCO/PoseTrack18 pretraining
is allowed. This release does not resolve that discrepancy or assert a COCO-only
warm start. **Organizer confirmation of starter-checkpoint eligibility is needed.**
No extra external images or pseudo-labels were introduced during our fine-tuning.

## 7. Fact sheet and handoff

`fact_sheet/main.pdf` + the LaTeX project in `fact_sheet/` document this method and
the supplied official score. Email them to **upar.challenge@gmail.com**, with
this repository URL and the Git LFS/release weight links, by **October 26, 2026
AoE (UTC-12)**, i.e. **October 27, 2026 12:00 UTC**. The email draft is in
`ORGANIZER_EMAIL.txt`. The source project ZIP is attached to the GitHub release.

```bash
# Rebuild the three handoff archives after compiling fact_sheet/main.tex:
python build_handoff.py
# Produces handoff/Track3_{source,weights,fact_sheet}.zip
```

Local verification evidence and its scope are summarized in
[`evidence/VERIFICATION.md`](evidence/VERIFICATION.md). For an optional offline
training/checkpoint backward smoke test in the training environment, run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python training/smoke_test.py
```

## 8. Attribution and licensing

See `THIRD_PARTY_NOTICES.md` and `licenses/` for starter-kit and adapted ProbPose
provenance. Upstream code/data/model licenses are not replaced by a blanket MIT
license. This is a research reproducibility bundle; dataset licenses still apply.
