#!/usr/bin/env python3
"""Patient-level 3-D TUMOR eval — whole-volume DSC + NSD@2mm + HD95 for the AUSAM tumor models.

The organ numbers are already 3-D (eval_ausam_3d.py); the tumor numbers were still 2-D on target-present
slices. This closes that gap: for each tumor-test patient (same seed-42 tumor split as training), run the
per-dataset tumor model over the tumor-present slices (semi-oracle GT box + "tumor" text), assemble the
3-D tumor mask at native resolution, and score it against the GT tumor volume — DSC, plus NSD@2mm / HD95
where the volume carries mm spacing (LiTS .npy does not, so LiTS is DSC-only, as for organs).

  python eval_tumor_3d.py --dataset msd|lits|kits|flare23 [--limit N]
-> results/tumor_3d_<dataset>.json
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, "/home/ud3d4/Desktop/SWOG/src/scripts")
from build_predicted_corpus import predict_volume, DEV
from eval_ausam_3d import load_msd, load_lits, load_kits, load_flare23, surface_metrics, NSD_TAUS, FFC
from run_pancreas_sam3 import _load_sam3_ckpt, SAM3_MODEL_ID, HF_TOKEN

TPOOL = "/scratch/ud3d4/acm_data/tumor_pool"
# dataset -> tumor model, tumor label(s) in the volume, loader, slice axis, tumor-split dataset key
CFG = {
    "msd":     {"tm": f"{TPOOL}/sam3_tumor_ausam_pancreas.pth", "tlab": [2],  "load": load_msd,     "ax": 2, "tds": "pancreas"},
    "lits":    {"tm": f"{TPOOL}/sam3_tumor_ausam_lits.pth",     "tlab": [2],  "load": load_lits,    "ax": 0, "tds": "lits"},
    "kits":    {"tm": f"{TPOOL}/sam3_tumor_ausam_kits.pth",     "tlab": [2],  "load": load_kits,    "ax": 0, "tds": "kits"},
    "flare23": {"tm": f"{TPOOL}/sam3_tumor_ausam_flare.pth",    "tlab": [14], "load": load_flare23, "ax": 2, "tds": "flare"},
}


def tumor_test_cases(ds, tds):
    """Same seed-42 tumor split used everywhere. msd/lits/kits are already materialized in the predicted
    corpus (fast); flare23 comes from the tumor split, intersected with the local FLARE23 volumes."""
    cp = f"/home/ud3d4/Desktop/SWOG/results/corpus_predicted_{ds}.json"
    if os.path.exists(cp):
        return [r["case_id"] for r in json.load(open(cp))["records"]]
    from train_tumor_incremental import patient_split as tumor_split, load_all as tumor_load_all
    Mt = tumor_load_all()[2]
    _, _, te = tumor_split(Mt)
    cases = sorted({Mt[i]["case"] for i in te if Mt[i]["dataset"] == tds})
    av = {os.path.basename(f).replace("_ct.nii.gz", "") for f in glob.glob(f"{FFC}/*_ct.nii.gz")}
    n0 = len(cases); cases = [c for c in cases if c in av]
    print(f"  (local FLARE23 volumes cover {len(cases)}/{n0} tumor-test patients)", flush=True)
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    cfg = CFG[a.dataset]
    from transformers import Sam3Processor
    proc = Sam3Processor.from_pretrained(SAM3_MODEL_ID, token=HF_TOKEN)
    tm = _load_sam3_ckpt(cfg["tm"], DEV)

    cases = tumor_test_cases(a.dataset, cfg["tds"])
    if a.limit:
        cases = cases[:a.limit]
    print(f"3-D TUMOR [{a.dataset}] {len(cases)} tumor-test patients", flush=True)

    dsc, nsdL, hdL, cout, skipped = [], [], [], [], 0
    for ci, case in enumerate(cases):
        try:
            ct, seg, sp = cfg["load"](case)
        except Exception as e:
            print(f"  [{ci+1}/{len(cases)}] {case} SKIP ({type(e).__name__}: {str(e)[:50]})", flush=True); continue
        gt = np.isin(seg, cfg["tlab"])
        if int(gt.sum()) == 0:                      # no GT tumor in this volume -> can't score tumor DSC
            skipped += 1; continue
        pred = predict_volume(tm, proc, ct, seg, cfg["tlab"], cfg["ax"], "tumor")
        inter = int((pred & gt).sum()); s = int(pred.sum()) + int(gt.sum())
        d = 2 * inter / s if s else 1.0
        dsc.append(d)
        rec = {"case": case, "dice3d": round(d, 4), "vox_pred": int(pred.sum()), "vox_gt": int(gt.sum())}
        if sp is not None:                          # surface metrics need mm spacing
            sm = surface_metrics(pred.astype(bool), gt.astype(bool), sp)
            rec["nsd3d"] = sm["nsd"]; rec["hd95_mm"] = sm["hd95_mm"]
            nsdL.append(sm["nsd"]["2.0mm"])
            if sm["hd95_mm"] is not None:
                hdL.append(sm["hd95_mm"])
        cout.append(rec)
        if (ci + 1) % 5 == 0 or ci == 0:
            print(f"  [{ci+1}/{len(cases)}] {case}: 3Ddice={round(d,4)} (pred {int(pred.sum())} / gt {int(gt.sum())} vox)", flush=True)

    out = {"dataset": a.dataset, "target": "tumor", "n_patients": len(cout), "n_skipped_no_gt_tumor": skipped,
           "nsd_taus_mm": list(NSD_TAUS),
           "mean_3d_dice": round(float(np.mean(dsc)), 4) if dsc else None,
           "mean_nsd_2mm": round(float(np.mean(nsdL)), 4) if nsdL else None,
           "mean_hd95_mm": round(float(np.mean(hdL)), 2) if hdL else None, "cases": cout}
    fp = f"/home/ud3d4/Desktop/SWOG/results/tumor_3d_{a.dataset}.json"
    json.dump(out, open(fp, "w"), indent=2)
    print(f"\n=== {a.dataset} TUMOR 3-D  DSC {out['mean_3d_dice']} | NSD@2mm {out['mean_nsd_2mm']} | "
          f"HD95mm {out['mean_hd95_mm']}  (n={len(cout)}, skipped {skipped}) ===", flush=True)
    print(f"-> {fp}", flush=True)


if __name__ == "__main__":
    main()
