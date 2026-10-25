# Inference only: same PyTorch/CUDA generation as the challenge server.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
WORKDIR /workspace
RUN python -m pip install --no-cache-dir numpy==2.4.6 Pillow==12.3.0
COPY . /workspace
ENV OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CMD ["python", "validate_release.py"]
