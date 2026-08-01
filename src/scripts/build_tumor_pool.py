#!/usr/bin/env python3
"""Pool tumor slices across datasets into ONE generic binary-tumor dataset (for a text-'tumor'
SAM3 model). LiTS liver-tumor + Pancreas pancreatic-tumor (images local). FLARE23 added later
(images not local yet). Each slice -> 256x256 uint8 RGB (HU-windowed) + binary tumor mask.
Patient/case tags kept so training can split without leakage.

out: /scratch/ud3d4/acm_data/tumor_pool/{images.npy, masks.npy, meta.json}
"""
import glob
import json
import os

import nibabel as nib
import numpy as np
from skimage.transform import resize

OUT = "/scratch/ud3d4/acm_data/tumor_pool"
os.makedirs(OUT, exist_ok=True)
LITS = "/scratch/ud3d4/acm_data/LiTS"
PANC = "/scratch/ud3d4/acm_data/Pancreas"


def hu_rgb(sl, lo, hi):
    x = np.clip(sl, lo, hi)
    x = ((x - lo) / (hi - lo) * 255).astype(np.uint8)
    return np.stack([x] * 3, -1)


imgs, masks, meta = [], [], []

# ---- LiTS (already sliced: grayscale [0,1] + binary tumor) ----
li = np.load(f"{LITS}/processed_images_tumor.npy")            # (7153,256,256) float [0,1]
ls = np.load(f"{LITS}/processed_segmentations_tumor.npy")     # (7153,256,256)
kept = 0
for i in range(len(ls)):
    if (ls[i] > 0).sum() < 50:
        continue
    g = (li[i] * 255).astype(np.uint8)
    imgs.append(np.stack([g] * 3, -1))
    masks.append((ls[i] > 0).astype(np.uint8))
    meta.append({"dataset": "lits", "case": f"lits_slice_{i}"})   # slice-level (no patient id in npy)
    kept += 1
print(f"LiTS: {kept} tumor slices", flush=True)

# ---- Pancreas (slice from NIfTI volumes; label 2 = tumor) ----
lo, hi = -100, 300
cases = sorted(glob.glob(f"{PANC}/labelsTr/pancreas_*.nii.gz"))
pk = 0
for lp in cases:
    cid = os.path.basename(lp).replace(".nii.gz", "")
    ip = f"{PANC}/imagesTr/{cid}.nii.gz"
    if not os.path.exists(ip):
        continue
    lab = np.asarray(nib.load(lp).dataobj).astype(np.uint8)
    if lab.ndim != 3 or (lab == 2).sum() < 15:
        continue
    vol = nib.load(ip).get_fdata()
    for z in range(lab.shape[2]):
        if (lab[:, :, z] == 2).sum() < 15:
            continue
        rgb = hu_rgb(resize(vol[:, :, z], (256, 256), preserve_range=True, anti_aliasing=True), lo, hi)
        m = resize((lab[:, :, z] == 2).astype(np.uint8), (256, 256), order=0, preserve_range=True).astype(np.uint8)
        imgs.append(rgb.astype(np.uint8))
        masks.append(m)
        meta.append({"dataset": "pancreas", "case": cid})
        pk += 1
print(f"Pancreas: {pk} tumor slices from {len(cases)} cases", flush=True)

X = np.stack(imgs).astype(np.uint8)
Y = np.stack(masks).astype(np.uint8)
np.save(f"{OUT}/images.npy", X)
np.save(f"{OUT}/masks.npy", Y)
json.dump(meta, open(f"{OUT}/meta.json", "w"))
from collections import Counter
print(f"POOL: {len(X)} slices  {X.shape} {X.dtype}  by dataset={dict(Counter(m['dataset'] for m in meta))}", flush=True)
print(f"-> {OUT}", flush=True)
