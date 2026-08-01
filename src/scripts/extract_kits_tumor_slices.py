#!/usr/bin/env python3
"""Add KiTS (kidney tumor, CT) to the generic tumor pool. Thread-pool over cases: download the small
segmentation first, skip the 226 MB image unless a tumor (label 2) is present, slice tumor regions,
discard the volume. Storage-safe. Checkpoints every 20 cases.

out: /scratch/ud3d4/acm_data/kits_tumor_pool/{images.npy, masks.npy, meta.json}
"""
import json
import os
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import nibabel as nib
import numpy as np
from skimage.transform import resize

OUT = "/scratch/ud3d4/acm_data/kits_tumor_pool"
os.makedirs(OUT, exist_ok=True)
IMG_URL = "https://huggingface.co/datasets/neheller/KiTS-Challenge-Imaging/resolve/main/images/case_{:05d}.nii.gz"
SEG_URL = "https://raw.githubusercontent.com/neheller/kits23/main/dataset/case_{:05d}/segmentation.nii.gz"
CASES = list(range(300)) + list(range(400, 589))          # 489 KiTS23 training cases
WIN, TUMOR, MINPX = (-125, 225), 2, 15
N_WORKERS = int(os.environ.get("KITS_WORKERS", "4"))
LIMIT = int(os.environ.get("KITS_LIMIT", "180"))
_lock = threading.Lock()
_imgs, _masks, _meta, _done = [], [], [], []


def hu_rgb(sl):
    lo, hi = WIN
    x = np.clip(sl, lo, hi)
    x = ((x - lo) / (hi - lo) * 255).astype(np.uint8)
    return np.stack([x] * 3, -1)


def fetch(url, path):
    urllib.request.urlretrieve(url, path)


def process(cn):
    seg_p = f"/dev/shm/_kits_{cn}_seg.nii.gz"
    img_p = f"/dev/shm/_kits_{cn}_img.nii.gz"
    try:
        fetch(SEG_URL.format(cn), seg_p)
        seg = np.asarray(nib.load(seg_p).dataobj).astype(np.uint8)
        if int((seg == TUMOR).sum()) < MINPX:
            return f"case_{cn:05d}: no tumor"
        fetch(IMG_URL.format(cn), img_p)
        ct = nib.load(img_p).get_fdata()
        if ct.shape != seg.shape:
            return f"case_{cn:05d}: shape mismatch"
        rows = []
        for z in range(seg.shape[0]):                     # KiTS axial = dim 0
            if int((seg[z] == TUMOR).sum()) < MINPX:
                continue
            rows.append((hu_rgb(resize(ct[z], (256, 256), preserve_range=True, anti_aliasing=True)).astype(np.uint8),
                         resize((seg[z] == TUMOR).astype(np.uint8), (256, 256), order=0, preserve_range=True).astype(np.uint8)))
        with _lock:
            for rgb, m in rows:
                _imgs.append(rgb); _masks.append(m); _meta.append({"dataset": "kits", "case": f"case_{cn:05d}"})
            _done.append(cn)
            n = len(_done)
            if n % 20 == 0:
                _save()
        return f"case_{cn:05d}: +{len(rows)} slices (pool {len(_imgs)}, {len(_done)} cases)"
    except Exception as e:
        return f"case_{cn:05d}: ERR {type(e).__name__}: {str(e)[:70]}"
    finally:
        for p in (seg_p, img_p):
            if os.path.exists(p):
                os.remove(p)


def _save():
    np.save(f"{OUT}/images.npy", np.stack(_imgs).astype(np.uint8))
    np.save(f"{OUT}/masks.npy", np.stack(_masks).astype(np.uint8))
    json.dump(_meta, open(f"{OUT}/meta.json", "w"))


def main():
    todo = CASES[:LIMIT]
    print(f"KiTS: {len(todo)} cases on {N_WORKERS} threads (tumor=label {TUMOR})", flush=True)
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        for i, msg in enumerate(ex.map(process, todo)):
            print(f"[{i+1}/{len(todo)}] {msg}", flush=True)
    if _imgs:
        with _lock:
            _save()
    print(f"DONE KiTS tumor pool: {len(_imgs)} slices from {len(_done)} cases -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
