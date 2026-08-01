#!/usr/bin/env python3
"""KiTS KIDNEY organ slices for the organ pool. Same thread-pooled, storage-safe pattern as the KiTS
tumor extractor, but slices label 1 (kidney) and tags organ='kidney'. Downloads seg (tiny) + image
(226 MB) per case, slices, discards the volume. Checkpoints every 20 cases.

out: /scratch/ud3d4/acm_data/kits_kidney_pool/{images.npy, masks.npy, meta.json}
"""
import json
import os
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import nibabel as nib
import numpy as np
from skimage.transform import resize

OUT = "/scratch/ud3d4/acm_data/kits_kidney_pool"
os.makedirs(OUT, exist_ok=True)
IMG_URL = "https://huggingface.co/datasets/neheller/KiTS-Challenge-Imaging/resolve/main/images/case_{:05d}.nii.gz"
SEG_URL = "https://raw.githubusercontent.com/neheller/kits23/main/dataset/case_{:05d}/segmentation.nii.gz"
CASES = list(range(300)) + list(range(400, 589))
WIN, KIDNEY, MINPX = (-125, 225), 1, 40
N_WORKERS = int(os.environ.get("KITS_WORKERS", "4"))
LIMIT = int(os.environ.get("KITS_LIMIT", "100"))          # ~100 cases -> ~6k kidney slices (balance)
STEP = int(os.environ.get("KITS_STEP", "2"))              # every other kidney slice
_lock = threading.Lock()
_imgs, _masks, _meta, _done = [], [], [], []


def hu_rgb(sl):
    lo, hi = WIN
    x = np.clip(sl, lo, hi)
    return np.stack([((x - lo) / (hi - lo) * 255).astype(np.uint8)] * 3, -1)


def _save():
    np.save(f"{OUT}/images.npy", np.stack(_imgs).astype(np.uint8))
    np.save(f"{OUT}/masks.npy", np.stack(_masks).astype(np.uint8))
    json.dump(_meta, open(f"{OUT}/meta.json", "w"))


def process(cn):
    seg_p, img_p = f"/dev/shm/_kk_{cn}_seg.nii.gz", f"/dev/shm/_kk_{cn}_img.nii.gz"
    try:
        urllib.request.urlretrieve(SEG_URL.format(cn), seg_p)
        seg = np.asarray(nib.load(seg_p).dataobj).astype(np.uint8)
        if int((seg == KIDNEY).sum()) < MINPX:
            return f"case_{cn:05d}: no kidney"
        urllib.request.urlretrieve(IMG_URL.format(cn), img_p)
        ct = nib.load(img_p).get_fdata()
        if ct.shape != seg.shape:
            return f"case_{cn:05d}: shape mismatch"
        rows = []
        for z in range(0, seg.shape[0], STEP):                 # KiTS axial = dim 0
            if int((seg[z] == KIDNEY).sum()) < MINPX:
                continue
            rows.append((hu_rgb(resize(ct[z], (256, 256), preserve_range=True, anti_aliasing=True)).astype(np.uint8),
                         resize((seg[z] == KIDNEY).astype(np.uint8), (256, 256), order=0, preserve_range=True).astype(np.uint8)))
        with _lock:
            for rgb, m in rows:
                _imgs.append(rgb); _masks.append(m); _meta.append({"dataset": "kits", "case": f"case_{cn:05d}", "organ": "kidney"})
            _done.append(cn)
            if len(_done) % 20 == 0:
                _save()
        return f"case_{cn:05d}: +{len(rows)} kidney slices (pool {len(_imgs)}, {len(_done)} cases)"
    except Exception as e:
        return f"case_{cn:05d}: ERR {type(e).__name__}: {str(e)[:60]}"
    finally:
        for p in (seg_p, img_p):
            if os.path.exists(p):
                os.remove(p)


def main():
    todo = CASES[:LIMIT]
    print(f"KiTS kidney: {len(todo)} cases on {N_WORKERS} threads", flush=True)
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        for i, msg in enumerate(ex.map(process, todo)):
            print(f"[{i+1}/{len(todo)}] {msg}", flush=True)
    if _imgs:
        with _lock:
            _save()
    print(f"DONE KiTS kidney pool: {len(_imgs)} slices from {len(_done)} cases -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
