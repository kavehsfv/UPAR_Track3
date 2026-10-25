"""Build organizer archives without data, credentials, caches or Git history.

Run from any directory after compiling fact_sheet/main.pdf. Weights are kept
separate so the lightweight source ZIP remains email-friendly.
"""
import argparse
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
WEIGHTS = ['assets/mipnet_ft.torchscript', 'weights/pose_model.pt']
EXCLUDED_DIRS = {'.git', '__pycache__', 'data', 'runs', 'submission_zip', 'handoff'}
EXCLUDED_SUFFIXES = {'.aux', '.bbl', '.blg', '.log', '.out', '.fls', '.gz', '.pyc', '.fdb_latexmk'}


def archive(output, members):
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in sorted(members):
            p = ROOT / name
            if not p.is_file():
                raise FileNotFoundError(p)
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 25, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            z.writestr(info, p.read_bytes())
    print(output)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', type=Path, default=ROOT / 'handoff')
    args = ap.parse_args()
    if not (ROOT / 'fact_sheet/main.pdf').is_file():
        raise FileNotFoundError('Compile fact_sheet/main.tex first')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in WEIGHTS:
        if (ROOT / name).stat().st_size < 1_000_000:
            raise RuntimeError(f'{name}: missing real weights; run git lfs pull')
    source = []
    for p in ROOT.rglob('*'):
        relative = p.relative_to(ROOT)
        if not p.is_file() or any(d in EXCLUDED_DIRS for d in relative.parts[:-1]):
            continue
        if p.suffix in EXCLUDED_SUFFIXES or str(relative) in WEIGHTS:
            continue
        source.append(relative.as_posix())
    fact = [n for n in source if n.startswith('fact_sheet/')]
    archive(args.out_dir / 'Track3_weights.zip', WEIGHTS)
    archive(args.out_dir / 'Track3_source.zip', source)
    archive(args.out_dir / 'Track3_fact_sheet.zip', fact)


if __name__ == '__main__':
    main()
