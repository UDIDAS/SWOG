#!/usr/bin/env python3
"""Rebuild the tumor_pool LiTS portion at PATIENT level: re-extract LiTS tumor slices from the 131 raw
per-patient volumes (case = volume-N), keep the existing Pancreas slices, discard the old slice-level
LiTS (fake `lits_slice_*` ids). Result: every LiTS slice carries a real patient id, so the tumor-model
split can hold out whole LiTS patients — zero slice-level data. Backs up the old pool first (reversible).
"""
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, "/home/ud3d4/Desktop/SWOG/src/scripts")
from run_lits_sam3 import extract_slices_caselevel, get_volume_ids

POOL = "/scratch/ud3d4/acm_data/tumor_pool"
BAK = "/scratch/ud3d4/acm_data/_archive_old_flare/tumor_pool_slicelevel_lits"  # reversible backup

# 1. back up the current pool
os.makedirs(BAK, exist_ok=True)
for f in ("images.npy", "masks.npy", "meta.json"):
    if not os.path.exists(f"{BAK}/{f}"):
        shutil.copy(f"{POOL}/{f}", f"{BAK}/{f}")
print("backed up old pool ->", BAK, flush=True)

# 2. keep only the Pancreas slices (already patient-level)
img = np.load(f"{POOL}/images.npy"); msk = np.load(f"{POOL}/masks.npy")
meta = json.load(open(f"{POOL}/meta.json"))
pan = [i for i, m in enumerate(meta) if m["dataset"] == "pancreas"]
print(f"kept pancreas slices: {len(pan)} (dropping {len(meta)-len(pan)} slice-level LiTS)", flush=True)
p_img, p_msk, p_meta = img[pan], msk[pan], [meta[i] for i in pan]

# 3. re-extract LiTS per patient volume, tagging case = volume-N
vol_ids = get_volume_ids()
print(f"extracting patient-level LiTS from {len(vol_ids)} volumes...", flush=True)
l_imgs, l_msks, l_meta = [], [], []
for k, vid in enumerate(vol_ids):
    xi, yi = extract_slices_caselevel([vid], label_value=2, threshold=50)
    if len(xi) == 0:
        continue
    l_imgs.append(xi); l_msks.append(yi)
    l_meta += [{"dataset": "lits", "case": f"volume-{vid}"}] * len(xi)
    if (k + 1) % 25 == 0:
        print(f"  {k+1}/{len(vol_ids)} volumes, {sum(len(a) for a in l_imgs)} LiTS slices", flush=True)
l_img = np.concatenate(l_imgs); l_msk = np.concatenate(l_msks)
print(f"patient-level LiTS: {len(l_img)} slices across {len(set(m['case'] for m in l_meta))} patients", flush=True)

# 4. combine + save
new_img = np.concatenate([p_img, l_img]); new_msk = np.concatenate([p_msk, l_msk])
new_meta = p_meta + l_meta
np.save(f"{POOL}/images.npy", new_img); np.save(f"{POOL}/masks.npy", new_msk)
json.dump(new_meta, open(f"{POOL}/meta.json", "w"))
import collections
print("NEW pool:", dict(collections.Counter(m["dataset"] for m in new_meta)),
      "| lits patients:", len({m["case"] for m in new_meta if m["dataset"] == "lits"}), flush=True)
