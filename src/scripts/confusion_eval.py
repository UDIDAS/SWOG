#!/usr/bin/env python3
"""Pixel-level confusion matrices for the trained models, on the 40 FLARE full cases (full 13-organ + tumor
GT — the only local data where every structure is labelled, so cross-structure confusion is measurable).

  ORGAN model : {liver, pancreas, kidney, others} x {liver, pancreas, kidney, others}  (symmetric 4-class;
                "others" = background + every non-target structure). Row-normalised = per-true-class recall
                on the diagonal, misses/false-fires off it.
  TUMOR model : {tumor, others} x {tumor, others}  (detection confusion), row-normalised.

Needs one GPU. Writes results/confusion_<tag>.json and heatmap PNGs. Parameterised so it can score any
checkpoint pair (baseline vs KG).  Usage:
  python confusion_eval.py [--organ-ckpt X.pth] [--tumor-ckpt Y.pth] [--tag baseline] [--n 40]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import nibabel as nib
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/ud3d4/Desktop/SWOG/src/scripts")
import infer_ensemble as IE
from run_pancreas_sam3 import _load_sam3_ckpt

CASES = "/scratch/ud3d4/acm_data/flare_full_cases"
DEF_ORGAN = "/scratch/ud3d4/acm_data/organ_pool_lkp/sam3_organ_generic.pth"
DEF_TUMOR = "/scratch/ud3d4/acm_data/tumor_pool/sam3_tumor_generic.pth"
R = "/home/ud3d4/Desktop/SWOG/results"
CLASSES = ["liver", "pancreas", "kidney", "others"]        # both axes; index: liver0 pancreas1 kidney2 others3
GT_MAP = {1: 0, 4: 1, 2: 2, 13: 2}                         # GT label -> class idx (else -> 3 others)
OTHERS = 3


def gt_to_class(gt):
    c = np.full(gt.shape, OTHERS, np.uint8)                # default "others" (bg + non-target structures)
    for lab, idx in GT_MAP.items():
        c[gt == lab] = idx
    return c


def organ_pred(ct, model, proc):
    p = np.full(ct.shape, OTHERS, np.uint8)                # default "others" (= nothing predicted here)
    for concept, cls in [("liver", 0), ("kidney", 2), ("pancreas", 1)]:  # pancreas last -> wins rare overlaps
        p[IE.seg_concept(ct, model, proc, concept) > 0] = cls
    return p


def heatmap(cm, classes, title, path):
    rn = cm / cm.sum(1, keepdims=True).clip(min=1)         # row-normalise (per true class)
    fig, ax = plt.subplots(figsize=(1.5 * len(classes) + 1.5, 1.3 * len(classes) + 1))
    im = ax.imshow(rn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{rn[i,j]:.2f}", ha="center", va="center",
                    color="white" if rn[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, fraction=0.046, pad=0.04); fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--organ-ckpt", default=DEF_ORGAN); ap.add_argument("--tumor-ckpt", default=DEF_TUMOR)
    ap.add_argument("--tag", default="baseline"); ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    _, proc = IE.load_base()
    organ = _load_sam3_ckpt(a.organ_ckpt, "cuda"); tumor = _load_sam3_ckpt(a.tumor_ckpt, "cuda")
    cids = sorted(os.path.basename(f)[:-len("_ct.nii.gz")] for f in glob.glob(f"{CASES}/*_ct.nii.gz"))[:a.n]
    print(f"confusion eval [{a.tag}] over {len(cids)} full FLARE cases", flush=True)

    ocm = np.zeros((4, 4), np.int64)                       # organ: liver/pancreas/kidney/others (symmetric)
    tcm = np.zeros((2, 2), np.int64)                       # tumor: tumor/others
    for k, cid in enumerate(cids):
        ct = nib.load(f"{CASES}/{cid}_ct.nii.gz").get_fdata()
        gt = np.asarray(nib.load(f"{CASES}/{cid}_label.nii.gz").dataobj).astype(np.uint8)
        gc = gt_to_class(gt); pc = organ_pred(ct, organ, proc)
        for ti in range(4):
            m = gc == ti
            if m.any():
                for pj in range(4):
                    ocm[ti, pj] += int((m & (pc == pj)).sum())
        tp = IE.seg_concept(ct, tumor, proc, "tumor") > 0; gt_t = gt == 14
        tcm[0, 0] += int((gt_t & tp).sum());   tcm[0, 1] += int((gt_t & ~tp).sum())     # true tumor
        tcm[1, 0] += int((~gt_t & tp).sum());  tcm[1, 1] += int((~gt_t & ~tp).sum())    # true others
        print(f"[{k+1}/{len(cids)}] {cid}", flush=True); torch.cuda.empty_cache()

    heatmap(ocm, CLASSES, f"Organ model — pixel confusion ({a.tag})", f"{R}/confusion_organ_{a.tag}.png")
    heatmap(tcm, ["tumor", "others"], f"Tumor model — pixel confusion ({a.tag})", f"{R}/confusion_tumor_{a.tag}.png")
    out = {"tag": a.tag, "n_cases": len(cids),
           "organ": {"classes": CLASSES, "counts": ocm.tolist()},
           "tumor": {"classes": ["tumor", "others"], "counts": tcm.tolist()}}
    json.dump(out, open(f"{R}/confusion_{a.tag}.json", "w"), indent=2)
    print(f"-> {R}/confusion_{a.tag}.json  + confusion_organ_{a.tag}.png / confusion_tumor_{a.tag}.png", flush=True)


if __name__ == "__main__":
    main()
