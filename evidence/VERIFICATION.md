# Verification record — final Track 3 entry

## Official result and participant confirmation

The participant confirms that the best/final method was uploaded to the challenge.
`leaderboard.csv` is the supplied organizer export for this final entry:

- Username: `kavehsfv-994308-994308_VID-Docker-GPU`
- Test set: 36,509
- AP (mean): **0.8862830000**
- AP50: **0.9672830000**
- AP_medium: **0.8564020000**
- AP_large: **0.6790190000**

These are the official evaluation results, not a newly computed local score.
No independent re-execution of the hidden challenge evaluation was performed
while preparing the release.

## Local checks passed

1. SHA256 of all six inference files and the training checkpoint checked against
   `artifacts.json`.
2. Rebuilt ZIP tested for integrity; all six inference file contents are
   byte-identical to the final runtime-fixed original archive
   (`task3_mipnet_v3.zip`, MD5 `2de661186c5954f38c8f2bd6cdd4fb96`).
3. Under **PyTorch 2.4.1+cpu**, offline inference passes the challenge output
   contract for 0, 1 and 9 boxes (including chunking) and a degenerate box.
   Outputs are finite, valid boxes yield bounded image coordinates and repeated
   crops are consistent. See `validation_cpu.txt`.
4. Full supplied training checkpoint loaded strictly: no missing/unexpected keys.
   Both frozen and unfrozen training stages produce finite losses and gradients;
   backbone gradients are absent/present as expected. See `training_smoke.txt`.
5. Offline export from the supplied state dictionary succeeds. Heatmaps sum to
   one per channel; variable batch sizes are supported. See `export_validation.txt`.
6. Re-exported and original deployment models give **zero maximum output
   difference** on the tested tensors at batch sizes 1, 6 and 12 in PyTorch
   2.4.1 CPU. See `export_parity.txt`.

## Scope and limitations

- These checks establish packaging, checkpoint/export consistency and local API
  compatibility; they are not a new measurement of hidden-set accuracy.
- Docker recipes are supplied, but Docker is unavailable on the preparation host;
  no successful container build/run is claimed.
- The complete training run was not repeated during release preparation. Original
  training history and selected epoch are supplied; numerical retraining may vary.
- Starter-checkpoint pretraining provenance and its eligibility clarification are
  disclosed in README section 6 and the fact sheet.
- Historical paper-tree metrics use a non-COCO reduction and are not represented
  as official challenge AP.
