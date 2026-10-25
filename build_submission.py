"""Build a deterministic, inference-only Codabench archive (no downloads)."""
import argparse
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
MEMBERS = ['run.py', 'metadata.yaml', 'requirements.txt', 'assets/mean_pose.json',
           'assets/preprocess_mip.json', 'assets/mipnet_ft.torchscript']

def build(output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in MEMBERS:
            path = ROOT / name
            if not path.is_file():
                raise FileNotFoundError(f'{path}: run git lfs pull first')
            if path.stat().st_size < 1000 and name.endswith('.torchscript'):
                raise RuntimeError(f'{path} is a Git LFS pointer, not weights')
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 16, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            z.writestr(info, path.read_bytes())
    print(output.resolve())
    return output

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', default=str(ROOT / 'submission_zip/task3_mipnet_v3.zip'))
    build(p.parse_args().out)
