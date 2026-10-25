"""Offline ingestion-contract checks, weight checksums and archive parity."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parent

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def check_output(result, boxes, w, h):
    assert len(result['keypoints']) == len(boxes)
    assert len(result['scores']) == len(boxes)
    if not boxes:
        return
    kp = np.asarray(result['keypoints'])
    assert kp.shape == (len(boxes), 17, 3) and np.isfinite(kp).all()
    assert np.isfinite(result['scores']).all()
    valid = [i for i,b in enumerate(boxes) if b[2] > 0 and b[3] > 0]
    assert ((kp[valid, :, 0] >= 0) & (kp[valid, :, 0] < w)).all()
    assert ((kp[valid, :, 1] >= 0) & (kp[valid, :, 1] < h)).all()
    assert np.isin(kp[:, :, 2], [0, 1, 2]).all()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--zip', type=Path)
    p.add_argument('--compare-original', type=Path)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    manifest = json.loads((ROOT / 'evidence/artifacts.json').read_text())
    for name, expected in manifest['sha256'].items():
        assert sha(ROOT / name) == expected, f'checksum mismatch: {name}'
    with tempfile.TemporaryDirectory() as td:
        folder = ROOT
        if args.zip:
            folder = Path(td) / 'submission'
            with zipfile.ZipFile(args.zip) as z:
                assert z.testzip() is None
                for n in z.namelist():
                    assert not Path(n).is_absolute() and '..' not in Path(n).parts
                z.extractall(folder)
            for name in manifest['inference_files']:
                assert sha(folder / name) == sha(ROOT / name), name
        if args.compare_original:
            with zipfile.ZipFile(args.compare_original) as z:
                for name in manifest['inference_files']:
                    assert hashlib.sha256(z.read(name)).hexdigest() == sha(ROOT / name), name
            print('Original archive inference payload: BYTE-IDENTICAL')
        spec = importlib.util.spec_from_file_location('release_run', folder / 'run.py')
        run = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run)
        run.load_model()
        assert run.TTA_SCALES == (0.9, 1.0, 1.1) and not run.FLIP_SHIFT
        rng = np.random.default_rng(17)
        im = Image.fromarray(rng.integers(0, 256, (256,192,3), dtype=np.uint8))
        path = Path(td) / 'sample.png'
        im.save(path)
        reports = []
        for count in [0, 1, 9]:
            boxes = [[5,5,180,245] for _ in range(count)]
            sample = dict(image_path=str(path), id=1, file_name=path.name,
                          width=192, height=256, boxes=boxes, box_ids=list(range(count)))
            output = run.predict_image(sample)
            check_output(output, boxes, 192, 256)
            if count == 9:
                np.testing.assert_allclose(output['keypoints'][0], output['keypoints'][-1], atol=2e-3)
            reports.append(dict(boxes=count, success=True))
        sample['boxes'] = [[5,5,0,20]]
        sample['box_ids'] = [0]
        check_output(run.predict_image(sample), sample['boxes'], 192, 256)
        print(json.dumps(dict(torch=torch.__version__, device=str(run._DEVICE),
                              checksums='passed', contract=reports, degenerate_box='passed'), indent=2))

if __name__ == '__main__':
    main()
