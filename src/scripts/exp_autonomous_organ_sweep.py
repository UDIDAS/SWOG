#!/usr/bin/env python3
"""Experiment C — autonomous per-organ Dice sweep vs the semi-oracle ceiling.

Fully autonomous organ segmentation (SAM3 concept prompt, NO box, NO label) on the 9 local held-out
FLARE cases, per organ, vs ground truth. Compares to the semi-oracle (GT-box) ceiling we already
measured. Base-SAM3 concept prompting is not trained per-case, so there is NO leakage — this is the
honest "what runs on an unlabeled scan" number. (The generic tumor model is excluded here: its FLARE
tumor slices are in the training pool, so tumor Dice on these cases would not be held-out; the honest
tumor number is the pooled val Dice 0.938.)
"""
import glob
import json
import os
import sys

import numpy as np
import nibabel as nib
import torch

sys.path.insert(0, "/home/ud3d4/Desktop/SWOG/src/scripts")
from infer_ensemble import load_base, seg_concept

DATA = "/home/ud3d4/Desktop/SWOG/data"
OUT = "/home/ud3d4/Desktop/SWOG/results"
# concept text, GT label id (FLARE scheme), measured semi-oracle (GT-box) ceiling
ORGANS = [("liver", 1, 0.973), ("right kidney", 2, 0.957), ("spleen", 3, 0.962),
          ("pancreas", 4, 0.882), ("left kidney", 13, 0.956)]


def dice(pred_bin, gt_bin):
    s = int(pred_bin.sum()) + int(gt_bin.sum())
    return 2 * int((pred_bin & gt_bin).sum()) / s if s else None


def main():
    os.makedirs(OUT, exist_ok=True)
    cids = sorted({os.path.basename(f).replace("_0000.nii.gz", "")
                   for f in glob.glob(f"{DATA}/FLARE23_*_0000.nii.gz")
                   if os.path.exists(f.replace("_0000.nii.gz", ".nii.gz"))})
    print(f"autonomous organ sweep over {len(cids)} held-out FLARE cases: {cids}", flush=True)
    model, proc = load_base()

    per_case = {}
    for cid in cids:
        ct = nib.load(f"{DATA}/{cid}_0000.nii.gz").get_fdata()
        gt = nib.load(f"{DATA}/{cid}.nii.gz").get_fdata().astype(np.uint8)
        row = {}
        for concept, lab, _ in ORGANS:
            if int((gt == lab).sum()) == 0:
                row[concept] = None            # organ not annotated in this case's GT -> not evaluable
                print(f"  {cid}  {concept:13s} (no GT for this organ — skipped)", flush=True)
                continue
            m = seg_concept(ct, model, proc, concept) > 0
            d = dice(m, gt == lab)
            row[concept] = None if d is None else round(d, 3)
            print(f"  {cid}  {concept:13s} Dice={row[concept]}", flush=True)
        per_case[cid] = row

    summary = {}
    for concept, lab, ceil in ORGANS:
        ds = [per_case[c][concept] for c in cids if per_case[c][concept] is not None]
        summary[concept] = {"autonomous_mean": round(float(np.mean(ds)), 3) if ds else None,
                            "n": len(ds), "semi_oracle_ceiling": ceil,
                            "gap": round(ceil - float(np.mean(ds)), 3) if ds else None}
        print(f"MEAN {concept:13s} autonomous={summary[concept]['autonomous_mean']} "
              f"ceiling={ceil} gap={summary[concept]['gap']} (n={len(ds)})", flush=True)

    json.dump({"summary": summary, "per_case": per_case}, open(f"{OUT}/autonomous_organ_sweep.json", "w"), indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [o[0] for o in ORGANS]
    auto = [summary[n]["autonomous_mean"] for n in names]
    ceil = [summary[n]["semi_oracle_ceiling"] for n in names]
    x = np.arange(len(names)); w = 0.38
    plt.figure(figsize=(8.5, 4.5))
    plt.bar(x - w / 2, auto, w, label="autonomous (concept prompt, no label)", color="#1e7a3c")
    plt.bar(x + w / 2, ceil, w, label="semi-oracle ceiling (GT box)", color="#b0b0b0")
    for i, v in enumerate(auto):
        if v is not None:
            plt.text(x[i] - w / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    plt.xticks(x, names, rotation=15); plt.ylim(0, 1.05)
    plt.ylabel("Dice"); plt.title("Autonomous organ segmentation vs semi-oracle ceiling (held-out FLARE)")
    plt.legend(); plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(f"{OUT}/autonomous_organ_sweep.png", dpi=130)
    print(f"-> {OUT}/autonomous_organ_sweep.json + .png", flush=True)


if __name__ == "__main__":
    main()
