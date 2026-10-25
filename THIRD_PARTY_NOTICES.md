# Attribution, licenses and pretrained-weight provenance

This research release adapts algorithms from **ProbPose** and uses the public
**ViTPose+ Small** model through Hugging Face Transformers. It is not an official
implementation or endorsement by those authors.

## ProbPose

- Miroslav Purkrabek and Jiri Matas, *ProbPose: A Probabilistic Approach to 2D
  Human Pose Estimation*, CVPR 2025, arXiv:2412.02254.
- Source studied: https://github.com/MiraPurkrabek/ProbPose_code .
- Upstream root LICENSE in the inspected checkout is **GNU GPL version 3**;
  the complete text is preserved in `licenses/ProbPose-GPL-3.0.txt` and `LICENSE`.
  The upstream `LICENSES.md` also describes inherited MMPose components; do not
  infer an Apache-only license for the adapted ProbPose method.
- Adapted portions: OKS-shaped targets, probability normalization/head,
  expected-OKS/Sobel loss and expected-OKS convolution decoding.
- Modifications by Kaveh Safavigerdini (2026): integration with a ViTPose+
  remapping head, self-contained PyTorch/NumPy operators, box-noise training,
  simplified presence branch, PT18 output, and stacked multi-scale/flip inference.
  GPLv3 is supplied for the adapted participant code; this does not relicense
  upstream libraries, datasets, model weights or organizer files.

## ViTPose / Transformers

- Yufei Xu, Jing Zhang, Qiming Zhang and Dacheng Tao, *ViTPose: Simple Vision
  Transformer Baselines for Human Pose Estimation*, NeurIPS 2022,
  arXiv:2204.12484. https://github.com/ViTAE-Transformer/ViTPose .
- Training dependency: Hugging Face Transformers, whose own license applies.
- Starter checkpoint: https://huggingface.co/usyd-community/vitpose-plus-small ,
  revision `0c30b6534bb621af0162b481176742577264e36e`.
  Model-card and upstream weight/data terms remain applicable to fine-tuned
  derivatives. No new license is asserted for the supplied pretrained-derived weights.
- **Eligibility disclosure:** expert route 0 does not erase shared multi-dataset
  pretraining. The model card names datasets beyond COCO/PoseTrack18; seek
  organizer confirmation of eligibility of their starter checkpoint.

## MIPNet and organizer starter

- MIPNet source studied: https://github.com/rawalkhirodkar/MIPNet .
  Its Apache-2.0 license is preserved in `licenses/MIPNet-Apache-2.0.txt`.
  This entry uses a locally designed box-jitter augmentation motivated by
  imperfect-box robustness; it does not include MIPNet's lambda-conditioned
  multi-instance architecture.
- Organizer starter interface and model choice:
  https://github.com/MickaelCormier/UPAR-Challenge-2027-Track3 .
  Organizer metadata/interface conventions remain subject to upstream terms.
- Data images, official hidden annotations and initial pretrained checkpoint
  files are not redistributed in this repository. Fine-tuned weights are
  supplied solely to make this research entry reproducible, subject to the
  applicable upstream model/dataset terms.

PyTorch, NumPy, Pillow, safetensors and huggingface-hub retain their respective
licenses. Refer to their installed package metadata for full notices.
