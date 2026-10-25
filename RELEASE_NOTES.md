# Track 3 final organizer handoff — v1.0

Final-method-only release for **kavehsfv-994308-994308_VID-Docker-GPU**.

Official participant-supplied leaderboard result: **AP 0.8862830000**, AP50
0.9672830000, AP_medium 0.8564020000, AP_large 0.6790190000 (test set 36,509).

## Downloads

- `Track3_fact_sheet.pdf`: compiled three-page fact sheet.
- `Track3_fact_sheet.zip`: complete LaTeX project, bibliography, class/style files
  and compiled PDF.
- `Track3_source.zip`: inference and training source, split/configuration,
  Dockerfiles, fact sheet and verification evidence. No dataset images or weights.
- `Track3_weights.zip`: `assets/mipnet_ft.torchscript` and
  `weights/pose_model.pt`; extract at the source/repository root.

Both weight files are also available through Git LFS in the repository.
Run `git lfs pull` after cloning. SHA256 checksums are recorded in
`evidence/artifacts.json`; `validate_release.py` verifies them locally.

## Verification scope

The inference payload is byte-identical to the final runtime-fixed submission.
Offline PyTorch 2.4.1 CPU checks pass for empty, single-box, chunked multi-box and
degenerate-box inputs. Strict training-checkpoint loading, frozen/unfrozen loss
and gradient checks, offline export, and deployment/export output parity pass.
These checks do not constitute a new hidden-server accuracy evaluation or full
retraining. Dockerfiles are supplied, but Docker was unavailable on the
preparation host, so container build/run is unverified.

## Required provenance disclosure

The supplied organizer-starter checkpoint is
`usyd-community/vitpose-plus-small`; its public model card describes multi-dataset
pretraining beyond COCO/PoseTrack18. Selecting expert 0 does not eliminate
supervision from shared parameters. The README and fact sheet disclose this
and request organizer confirmation of starter-checkpoint eligibility. No extra
external images or pseudo-labels were added during fine-tuning.

The email draft is in `ORGANIZER_EMAIL.txt`; publication of this release does
not send that email to the organizers.
