#!/usr/bin/env python3
"""KG train-split prior for the TUMOR model. Unlike organs, a tumor's LOCATION is not stable (it can sit in
liver, kidney or pancreas), so we use a SIZE-plausibility band only (centroid disabled in the loss). Computed
from TRAINING slices ONLY (same seed-42 patient split as train_tumor_incremental) → leakage-free. The band
discourages anatomically implausible tumor area (mainly over-segmentation) during KG-in-training.

out: /scratch/ud3d4/acm_data/tumor_pool/tumor_train_priors.json  → {"tumor": {cy,cx,area_lo,area_hi,...}}
"""
import json

import numpy as np

from train_tumor_incremental import POOLS, patient_split

Ys, M = [], []
for p in POOLS.values():
    Ys.append(np.load(f"{p}/masks.npy")); M += json.load(open(f"{p}/meta.json"))
Y = np.concatenate(Ys)
tr, va, te = patient_split(M)
H, W = Y.shape[1], Y.shape[2]
yy = (np.arange(H)[:, None] + 0.5) / H
xx = (np.arange(W)[None, :] + 0.5) / W

areas, cys, cxs = [], [], []
for i in tr:                                       # TRAIN split only -> leakage-free
    m = Y[i] > 0
    s = int(m.sum())
    if s < 1:
        continue
    areas.append(s / (H * W))
    cys.append(float((m * yy).sum() / s)); cxs.append(float((m * xx).sum() / s))

a = np.array(areas)
prior = {"tumor": {
    "cy": round(float(np.mean(cys)), 4), "cx": round(float(np.mean(cxs)), 4),   # kept but unused (centroid_w=0)
    "area_mean": round(float(np.mean(a)), 6),
    "area_lo": round(float(np.percentile(a, 5)), 6),
    "area_hi": round(float(np.percentile(a, 95)), 6),
    "n": int(len(a))}}
out = f"{POOLS['tumor_pool'].rsplit('/', 1)[0]}/tumor_pool/tumor_train_priors.json"
json.dump(prior, open(out, "w"), indent=2)
p = prior["tumor"]
print(f"tumor size-plausibility band (train slices only): area {p['area_lo']:.5f}..{p['area_hi']:.5f} "
      f"(mean {p['area_mean']:.5f}), n={p['n']}  [centroid disabled in loss]")
print(f"-> {out}")
