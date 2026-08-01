#!/usr/bin/env python3
"""Build the ORGAN concept-training pool (liver/kidney/pancreas) — the organ analogue of the tumor
pool. Each slice -> 256x256 uint8 RGB (HU-windowed) + binary organ mask + the organ name (the text
prompt used at train/inference). This file does the LOCAL sources (liver from LiTS, pancreas from MSD
Pancreas); kidney (KiTS) and FLARE diversity are added by the extractor scripts, then merged.

out: /scratch/ud3d4/acm_data/organ_pool/{images.npy, masks.npy, meta.json}
"""
import glob
import json
import os

import nibabel as nib
import numpy as np
from skimage.transform import resize

OUT = "/scratch/ud3d4/acm_data/organ_pool"
os.makedirs(OUT, exist_ok=True)
WIN = (-125, 225)
MINPX = 40
CAP_PER_ORGAN = 6000        # keep organs balanced with the ~5-7k tumor sources


def hu_rgb(sl):
    lo, hi = WIN
    x = np.clip(sl, lo, hi)
    return np.stack([((x - lo) / (hi - lo) * 255).astype(np.uint8)] * 3, -1)


imgs, masks, meta = [], [], []


def add_slice(ct2d, m2d, dataset, case, organ):
    imgs.append(hu_rgb(resize(ct2d, (256, 256), preserve_range=True, anti_aliasing=True)).astype(np.uint8))
    masks.append(resize(m2d.astype(np.uint8), (256, 256), order=0, preserve_range=True).astype(np.uint8))
    meta.append({"dataset": dataset, "case": case, "organ": organ})


# ---- LIVER from LiTS (label >=1 = liver incl. its tumor) ----
LITS = "/scratch/ud3d4/acm_data/Data"
vols = sorted(glob.glob(f"{LITS}/ct/volume-*.npy"))
n = 0
for vp in vols:
    if n >= CAP_PER_ORGAN:
        break
    sp = vp.replace("/ct/volume-", "/seg/segmentation-")
    if not os.path.exists(sp):
        continue
    ct = np.load(vp, mmap_mode="r"); seg = np.load(sp, mmap_mode="r")
    ax = 2 if ct.shape[2] < ct.shape[0] else 0                 # axial dim (LiTS npy usually (H,W,Z))
    Z = ct.shape[ax]
    for z in range(0, Z, 2):                                   # every other slice
        sl = ct[:, :, z] if ax == 2 else ct[z]
        ms = (seg[:, :, z] if ax == 2 else seg[z]) >= 1
        if int(ms.sum()) < MINPX:
            continue
        add_slice(np.asarray(sl), np.asarray(ms), "lits", os.path.basename(vp)[:-4], "liver")
        n += 1
        if n >= CAP_PER_ORGAN:
            break
print(f"LIVER (LiTS): {n} slices", flush=True)

# ---- PANCREAS from MSD Pancreas (label >=1 = pancreas incl. its tumor) ----
PANC = "/scratch/ud3d4/acm_data/Pancreas"
cases = sorted(glob.glob(f"{PANC}/labelsTr/pancreas_*.nii.gz"))
n = 0
for lp in cases:
    if n >= CAP_PER_ORGAN:
        break
    cid = os.path.basename(lp).replace(".nii.gz", "")
    ip = f"{PANC}/imagesTr/{cid}.nii.gz"
    if not os.path.exists(ip):
        continue
    lab = np.asarray(nib.load(lp).dataobj).astype(np.uint8)
    if lab.ndim != 3:
        continue
    vol = nib.load(ip).get_fdata()
    for z in range(lab.shape[2]):
        ms = lab[:, :, z] >= 1
        if int(ms.sum()) < MINPX:
            continue
        add_slice(vol[:, :, z], ms, "pancreas", cid, "pancreas")
        n += 1
        if n >= CAP_PER_ORGAN:
            break
print(f"PANCREAS (MSD): {n} slices", flush=True)

np.save(f"{OUT}/images.npy", np.stack(imgs).astype(np.uint8))
np.save(f"{OUT}/masks.npy", np.stack(masks).astype(np.uint8))
json.dump(meta, open(f"{OUT}/meta.json", "w"))
from collections import Counter
print(f"ORGAN pool (local): {len(imgs)} slices  {dict(Counter(m['organ'] for m in meta))} -> {OUT}", flush=True)
print("kidney (KiTS) + FLARE diversity added by the extractor scripts, then merged.", flush=True)
