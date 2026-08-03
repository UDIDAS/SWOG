#!/usr/bin/env python3
"""KG-guided segmentation — the KG helping segmentation (closing the loop).

`repair(mask, spacing)` applies the KG anatomical atlas (kg/data/kg_atlas.json) to an autonomous
multi-label mask:
  1. keep the plausible connected component per organ, drop spurious blobs (organs are single structures;
     kidneys are two separate labels);
  2. flag organs whose volume falls outside the cohort's plausible range (p1..p99);
  3. enforce tumor ⊂ organ — remove predicted tumor voxels that lie outside any organ.

This is CPU-only and runs as the last stage of the autonomous pipeline (infer_ensemble). The prompt-
guidance half (use atlas location/size to steer the concept prompt) is applied at inference on the GPU.

Run with no args for a self-check: it injects realistic errors into a real FLARE case's GT mask and shows
the repair recovers the Dice.
"""
import json
import os
import sys

import numpy as np
from scipy import ndimage

ATLAS = "/home/ud3d4/Desktop/SWOG/kg/data/kg_atlas.json"
# FLARE / KG multi-label scheme
ORGAN_LABELS = {"liver": 1, "right_kidney": 2, "spleen": 3, "pancreas": 4, "left_kidney": 13}
TUMOR = 14


def load_atlas(path=ATLAS):
    return json.load(open(path))["organs"]


def repair(mask, spacing, atlas=None, verbose=False):
    """Return (repaired_mask, report[]). Non-destructive to plausible structures."""
    atlas = atlas or load_atlas()
    vox_cm3 = float(np.prod(spacing)) / 1000.0
    out = mask.copy()
    report = []
    for name, lab in ORGAN_LABELS.items():
        m = mask == lab
        if not m.any():
            continue
        lbl, n = ndimage.label(m)
        if n > 1:                                   # organ = single connected structure -> keep the largest
            sizes = ndimage.sum(m, lbl, range(1, n + 1))
            keep = int(np.argmax(sizes)) + 1
            drop = m & (lbl != keep)
            out[drop] = 0
            report.append(f"{name}: dropped {n-1} spurious component(s) ({int(drop.sum())} vox)")
        vol = float((out == lab).sum()) * vox_cm3
        rng = atlas.get(name, {}).get("volume_cm3")
        if rng and (vol < rng["p1"] or vol > rng["p99"]):
            report.append(f"{name}: volume {vol:.0f} cm3 OUTSIDE plausible [{rng['p1']}..{rng['p99']}] — FLAG")
    # tumor must be ADJACENT to an organ (tumor is a separate label that carves out of the organ, so a
    # real tumor touches its host organ; a floating tumor component is spurious). Component-based.
    tm = out == TUMOR
    if tm.any():
        organ_dil = ndimage.binary_dilation(np.isin(out, list(ORGAN_LABELS.values())), iterations=3)
        lbl, n = ndimage.label(tm)
        removed = 0
        for c in range(1, n + 1):
            comp = lbl == c
            if not (comp & organ_dil).any():        # touches no organ -> spurious, drop
                out[comp] = 0; removed += int(comp.sum())
        if removed:
            report.append(f"tumor: removed {removed} vox in component(s) not adjacent to any organ")
    if verbose:
        for r in report:
            print("  -", r, flush=True)
    return out, report


def dice(a, b):
    s = a.sum() + b.sum()
    return 2.0 * np.logical_and(a, b).sum() / s if s else 1.0


def _self_check():
    import glob
    import nibabel as nib
    cases = sorted(glob.glob("/scratch/ud3d4/acm_data/flare_full_cases/*_label.nii.gz"))
    if not cases:
        print("no full-case labels found for self-check"); return
    nii = nib.load(cases[0]); gt = np.asarray(nii.dataobj).astype(np.uint8)
    sp = [float(z) for z in nii.header.get_zooms()[:3]]
    rng = np.random.RandomState(0)
    corrupt = gt.copy()
    # realistic autonomous errors: (a) a large spurious 'pancreas' blob away from the real pancreas,
    # (b) several floating 'tumor' blobs not attached to any organ.
    pv = int((gt == 4).sum())
    z, y, x = [rng.randint(30, s - 30) for s in gt.shape]
    corrupt[z-10:z+10, y-12:y+12, x-12:x+12] = 4                      # bogus pancreas blob (a 2nd component)
    for _ in range(6):
        z, y, x = [rng.randint(20, s - 20) for s in gt.shape]
        if not np.isin(gt[z, y, x], [1, 2, 3, 4, 13, 14]):
            corrupt[z-3:z+3, y-3:y+3, x-3:x+3] = TUMOR               # floating tumor blob (spurious)
    fixed, rep = repair(corrupt, sp, verbose=True)
    cid = os.path.basename(cases[0]).replace("_label.nii.gz", "")
    print(f"\ncase {cid}  ({gt.shape}, spacing {np.round(sp,2)})   Dice vs ground truth:")
    for lab, name in [(4, "pancreas"), (TUMOR, "tumor")]:
        print(f"  {name:8s} corrupted={dice(corrupt==lab, gt==lab):.3f}  ->  repaired={dice(fixed==lab, gt==lab):.3f}")


if __name__ == "__main__":
    _self_check()
