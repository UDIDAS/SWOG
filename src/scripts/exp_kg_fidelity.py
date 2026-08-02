#!/usr/bin/env python3
"""Experiment B1 — predicted-KG vs GT-KG answer fidelity.

Question: when the KG is built from AUTONOMOUS (label-free) segmentation instead of ground truth, does
it give the SAME answers? We have 20 paired pancreas cases (GT mask + autonomous prediction, identical
geometry). We derive the exact phenotypes the KG stores (volume, max-diameter, centroid, size bin) from
BOTH masks, then compare at three levels:
  ① node fidelity   — per-phenotype error + correlation (pred vs GT)
  ② categorical      — does the KG assign the same size bin?
  ③ query fidelity   — "rank patients by pancreas size": Spearman + top-k overlap of the two KGs' answers
Thread-pooled mask loading. CPU, seconds.
"""
import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import nibabel as nib
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

PD = "/scratch/ud3d4/acm_data/FLARE_Task2/sam3_delivery/pancreas"
OUT = "/home/ud3d4/Desktop/SWOG/results"
WORKERS = 8


def feret_mm(mask, sp):
    idx = np.argwhere(mask)
    if len(idx) < 2:
        return 0.0
    pts = idx * np.asarray(sp)
    if len(pts) > 3000:
        try:
            pts = pts[ConvexHull(pts).vertices]
        except Exception:
            pts = pts[np.random.RandomState(0).choice(len(pts), 3000, replace=False)]
    return float(pdist(pts).max())


def phen(mask, sp):
    """Phenotypes the KG stores, from a binary mask."""
    vox = int(mask.sum())
    vol = vox * float(np.prod(sp)) / 1000.0                 # cm^3
    cen = (np.argwhere(mask).mean(0) * np.asarray(sp)).tolist() if vox else [0, 0, 0]
    return {"volume_cm3": vol, "voxels": vox, "max_diameter_mm": feret_mm(mask, sp), "centroid_mm": cen}


def dice(a, b):
    s = a.sum() + b.sum()
    return float(2 * np.logical_and(a, b).sum() / s) if s else 1.0


def one_case(cid):
    gt = nib.load(f"{PD}/{cid}_gt.nii.gz"); pr = nib.load(f"{PD}/{cid}_pred.nii.gz")
    g = np.asarray(gt.dataobj) > 0; p = np.asarray(pr.dataobj) > 0
    sp = gt.header.get_zooms()[:3]
    return {"case": cid, "dice": dice(p, g), "gt": phen(g, sp), "pred": phen(p, sp)}


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def mape(pred, gt):
    gt = np.asarray(gt); pred = np.asarray(pred)
    return float(np.mean(np.abs(pred - gt) / (np.abs(gt) + 1e-9)) * 100)


def main():
    os.makedirs(OUT, exist_ok=True)
    cases = sorted(set(os.path.basename(f)[:-len("_pred.nii.gz")] for f in glob.glob(f"{PD}/*_pred.nii.gz")))
    print(f"paired pancreas cases: {len(cases)}", flush=True)
    with ThreadPoolExecutor(WORKERS) as ex:
        R = list(ex.map(one_case, cases))

    gv = np.array([r["gt"]["volume_cm3"] for r in R]); pv = np.array([r["pred"]["volume_cm3"] for r in R])
    gd = np.array([r["gt"]["max_diameter_mm"] for r in R]); pd_ = np.array([r["pred"]["max_diameter_mm"] for r in R])
    gc = np.array([r["gt"]["centroid_mm"] for r in R]); pc = np.array([r["pred"]["centroid_mm"] for r in R])
    dices = np.array([r["dice"] for r in R])

    # ③ query fidelity: "rank by pancreas size"
    order_gt = list(np.argsort(-gv)); order_pr = list(np.argsort(-pv))
    def topk_overlap(k):
        return len(set(order_gt[:k]) & set(order_pr[:k])) / k

    # ② categorical: tertile size bins defined on the GT distribution, applied to both
    q = np.quantile(gv, [1 / 3, 2 / 3])
    binf = lambda v: np.digitize(v, q)
    cat_agree = float(np.mean(binf(gv) == binf(pv)))

    res = {
        "n_cases": len(cases),
        "mean_dice": round(float(dices.mean()), 3),
        "node_fidelity": {
            "volume_MAPE_%": round(mape(pv, gv), 1), "volume_pearson_r": round(pearson(pv, gv), 3),
            "diameter_MAPE_%": round(mape(pd_, gd), 1), "diameter_pearson_r": round(pearson(pd_, gd), 3),
            "centroid_mean_mm": round(float(np.mean(np.linalg.norm(pc - gc, axis=1))), 1),
        },
        "categorical_size_bin_agreement": round(cat_agree, 3),
        "query_rank_by_size": {
            "spearman": round(spearman(pv, gv), 3),
            "top3_overlap": round(topk_overlap(3), 3), "top5_overlap": round(topk_overlap(5), 3),
        },
    }
    print(json.dumps(res, indent=2), flush=True)
    json.dump({"summary": res, "per_case": R}, open(f"{OUT}/kg_fidelity.json", "w"), indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for a, (x, y, lab, r) in zip(ax, [(gv, pv, "pancreas volume (cm³)", res["node_fidelity"]["volume_pearson_r"]),
                                       (gd, pd_, "pancreas max-diameter (mm)", res["node_fidelity"]["diameter_pearson_r"])]):
        lim = [0, max(x.max(), y.max()) * 1.05]
        a.plot(lim, lim, "--", color="gray", lw=1)
        a.scatter(x, y, c=dices, cmap="viridis", vmin=0.5, vmax=1.0, s=45, edgecolor="k", linewidth=0.3)
        a.set_xlabel(f"GT-KG  {lab}"); a.set_ylabel(f"predicted-KG  {lab}")
        a.set_title(f"{lab.split(' (')[0]}   r={r:.3f}"); a.set_xlim(lim); a.set_ylim(lim)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0.5, 1.0)); sm.set_array([])
    fig.colorbar(sm, ax=ax, label="per-case Dice", fraction=0.03)
    fig.suptitle(f"Predicted-KG vs GT-KG fidelity — pancreas, {len(cases)} cases (mean Dice {res['mean_dice']})",
                 fontweight="bold")
    plt.savefig(f"{OUT}/kg_fidelity.png", dpi=130, bbox_inches="tight")
    print(f"-> {OUT}/kg_fidelity.json + kg_fidelity.png", flush=True)


if __name__ == "__main__":
    main()
