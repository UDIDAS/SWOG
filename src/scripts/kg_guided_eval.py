#!/usr/bin/env python3
"""Quantify KG-guided segmentation: does the KG atlas repair improve autonomous masks?

For each full FLARE case: run fully-autonomous segmentation (SAM3 concept organs + generic tumor model,
no boxes) -> RAW mask; apply the KG-guided repair -> REPAIRED mask; score both per structure against GT.
Reports mean Dice raw vs repaired. GPU inference; writes results/kg_guided_eval.json.
"""
import glob
import json
import os
import sys

import numpy as np
import nibabel as nib
import torch

sys.path.insert(0, "/home/ud3d4/Desktop/SWOG/src/scripts")
import infer_ensemble as IE
from run_pancreas_sam3 import _load_sam3_ckpt
from kg_guided_segment import repair, ORGAN_LABELS, TUMOR

CASES = "/scratch/ud3d4/acm_data/flare_full_cases"
TUMOR_CKPT = "/scratch/ud3d4/acm_data/tumor_pool/sam3_tumor_generic.pth"   # final all-4-datasets model
OUT = "/home/ud3d4/Desktop/SWOG/results/kg_guided_eval.json"
STRUCTS = [("liver", 1), ("right_kidney", 2), ("spleen", 3), ("pancreas", 4), ("left_kidney", 13), ("tumor", TUMOR)]


def dice(a, b):
    s = a.sum() + b.sum()
    return float(2 * np.logical_and(a, b).sum() / s) if s else None


def main():
    cids = sorted(os.path.basename(f)[:-len("_ct.nii.gz")] for f in glob.glob(f"{CASES}/*_ct.nii.gz"))
    n = int(sys.argv[1]) if len(sys.argv) > 1 else len(cids)
    cids = cids[:n]
    print(f"KG-guided eval over {len(cids)} full FLARE cases", flush=True)
    model, proc = IE.load_base()
    tmodel = _load_sam3_ckpt(TUMOR_CKPT, "cuda")

    raw_acc, rep_acc = {s: [] for s, _ in STRUCTS}, {s: [] for s, _ in STRUCTS}
    for k, cid in enumerate(cids):
        ct = nib.load(f"{CASES}/{cid}_ct.nii.gz").get_fdata()
        nii = nib.load(f"{CASES}/{cid}_label.nii.gz"); gt = np.asarray(nii.dataobj).astype(np.uint8)
        sp = nii.header.get_zooms()[:3]
        pred = np.zeros(ct.shape, np.uint8)
        for concept, lab in IE.ORGANS:                       # autonomous concept organs (base SAM3)
            m = IE.seg_concept(ct, model, proc, concept)
            pred[m > 0] = lab
        tm = IE.seg_concept(ct, tmodel, proc, "tumor")       # autonomous tumor
        pred[(tm > 0) & (pred > 0)] = TUMOR                  # baseline pipeline: tumor kept inside an organ
        rep, _ = repair(pred, sp)                            # KG-guided repair
        row = []
        for s, lab in STRUCTS:
            g = gt == lab
            if g.sum() == 0:
                continue
            dr, dp = dice(pred == lab, g), dice(rep == lab, g)
            raw_acc[s].append(dr); rep_acc[s].append(dp); row.append(f"{s} {dr:.2f}->{dp:.2f}")
        print(f"[{k+1}/{len(cids)}] {cid}: " + "  ".join(row), flush=True)
        torch.cuda.empty_cache()

    summary = {}
    for s, _ in STRUCTS:
        if raw_acc[s]:
            summary[s] = {"n": len(raw_acc[s]),
                          "raw": round(float(np.mean(raw_acc[s])), 3),
                          "kg_repaired": round(float(np.mean(rep_acc[s])), 3)}
    json.dump({"summary": summary}, open(OUT, "w"), indent=2)
    print("\n=== KG-guided repair: mean Dice (raw -> repaired) ===", flush=True)
    for s, v in summary.items():
        print(f"  {s:13s} raw {v['raw']:.3f}  ->  KG-repaired {v['kg_repaired']:.3f}   (n={v['n']})", flush=True)
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
