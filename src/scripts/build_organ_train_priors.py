#!/usr/bin/env python3
"""KG train-split prior atlas for the ORGAN model — the anatomical knowledge the KG injects into training.
For each organ, compute (from TRAINING slices ONLY, same seed-42 patient split as the trainer, so it is
leakage-free) the normalized 2D centroid (where that organ typically sits in a 256x256 slice) and the
plausible area-fraction band (how much of the slice it typically occupies). These population priors drive
a differentiable plausibility/consistency loss during training (see kg_consistency_loss in run_pancreas_sam3).

out: /scratch/ud3d4/acm_data/organ_pool_lkp/organ_train_priors.json
"""
import json
from collections import defaultdict

import numpy as np

from train_organ_generic import patient_split, POOL

Y = np.load(f"{POOL}/masks.npy"); M = json.load(open(f"{POOL}/meta.json"))
tr, va, te = patient_split(M)
H, W = Y.shape[1], Y.shape[2]
yy = (np.arange(H)[:, None] + 0.5) / H          # normalized row coords
xx = (np.arange(W)[None, :] + 0.5) / W          # normalized col coords

acc = defaultdict(lambda: {"area": [], "cy": [], "cx": []})
for i in tr:                                      # TRAIN split only -> leakage-safe
    m = Y[i] > 0
    s = int(m.sum())
    if s < 1:
        continue
    o = M[i]["organ"]
    acc[o]["area"].append(s / (H * W))
    acc[o]["cy"].append(float((m * yy).sum() / s))
    acc[o]["cx"].append(float((m * xx).sum() / s))

priors = {}
for o, d in acc.items():
    a = np.array(d["area"])
    priors[o] = {
        "cy": round(float(np.mean(d["cy"])), 4),
        "cx": round(float(np.mean(d["cx"])), 4),
        "area_mean": round(float(np.mean(a)), 5),
        "area_lo": round(float(np.percentile(a, 5)), 5),    # plausible band (5-95 pct of training slices)
        "area_hi": round(float(np.percentile(a, 95)), 5),
        "n": int(len(a)),
    }

out = f"{POOL}/organ_train_priors.json"
json.dump(priors, open(out, "w"), indent=2)
print("KG train-split organ priors (leakage-free, train slices only):")
for o, p in sorted(priors.items()):
    print(f"  {o:9s} centroid=({p['cy']:.3f},{p['cx']:.3f})  area {p['area_lo']:.4f}..{p['area_hi']:.4f} "
          f"(mean {p['area_mean']:.4f})  n={p['n']}")
print(f"-> {out}")
