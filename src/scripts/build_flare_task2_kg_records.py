#!/usr/bin/env python3
"""Add the FLARE-Task2 organ set to the knowledge graph — the 100 fully-labelled organ volumes we also
train on. Labels are LOCAL (no download): 50 train + 50 public-val, 13-organ FLARE scheme, no tumor.
Computes per-patient organ phenotypes (volume, Feret diameter, centroid) and merges 'flare_task2' records
into the unified KG corpora (corpus_perpatient.json + corpus_global.json); writes corpus_flare_task2.json.
"""
import glob
import json
import os

import numpy as np
import nibabel as nib
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

D = "/home/ud3d4/Desktop/SWOG/kg/data"
LAB_DIRS = ["/scratch/ud3d4/acm_data/FLARE_Task2/train_gt_label/labelsTr",
            "/scratch/ud3d4/acm_data/FLARE_Task2/validation/Validation-Public-Labels"]
FLARE_ORGANS = {1: "liver", 2: "right_kidney", 3: "spleen", 4: "pancreas", 5: "aorta",
                6: "inferior_vena_cava", 7: "right_adrenal", 8: "left_adrenal", 9: "gallbladder",
                10: "esophagus", 11: "stomach", 12: "duodenum", 13: "left_kidney"}
MINVOX = 50


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
    return round(float(pdist(pts).max()), 1)


def one_case(lp):
    cid = os.path.basename(lp).replace(".nii.gz", "")
    nii = nib.load(lp); seg = np.asarray(nii.dataobj).astype(np.uint8)
    sp = [float(z) for z in nii.header.get_zooms()[:3]]
    vox = float(np.prod(sp)) / 1000.0                                   # cm^3 per voxel
    organs, observed = {}, []
    for lab, name in FLARE_ORGANS.items():
        m = seg == lab
        n = int(m.sum())
        if n < MINVOX:
            continue
        observed.append(name)
        organs[name] = {
            "present": True,
            "organ_volume_cm3": round(n * vox, 2),
            "organ_voxels": n,
            "organ_max_diameter_mm": feret_mm(m, sp),
            "organ_centroid_mm": [round(float(c), 1) for c in (np.argwhere(m).mean(0) * sp)],
            "has_tumor": False, "tumor_volume_cm3": 0.0, "tumor_voxels": 0,
            "tumor_max_diameter_mm": 0.0, "tumor_centroid_mm": None, "lesion_count": 0,
            "burden_cat": "none", "multiplicity": "none", "containment": "na",
            "anatomic_location": "na", "size_cat": "unknown"}
    if not observed:
        return None
    return {"dataset": "flare_task2", "case_id": cid, "granularity": "volume",
            "observed_organs": observed, "n_observed": len(observed), "organs": organs}


def main():
    files = sorted(f for d in LAB_DIRS for f in glob.glob(f"{d}/*.nii.gz"))
    print(f"phenotyping {len(files)} FLARE-Task2 organ volumes (local labels)...", flush=True)
    recs = []
    for lp in files:
        try:
            r = one_case(lp)
            if r:
                recs.append(r)
        except Exception as e:
            print(f"  {os.path.basename(lp)} ERR {type(e).__name__}: {str(e)[:60]}", flush=True)
    import collections
    oc = collections.Counter(o for r in recs for o in r["observed_organs"])
    print(f"FLARE-Task2 KG records: {len(recs)} patients | organ coverage: {dict(oc)}", flush=True)
    json.dump({"records": recs}, open(f"{D}/corpus_flare_task2.json", "w"))

    for name in ["corpus_perpatient.json", "corpus_global.json"]:
        obj = json.load(open(f"{D}/{name}"))
        keep = [r for r in obj["records"] if r.get("dataset") != "flare_task2"]
        obj["records"] = keep + recs
        json.dump(obj, open(f"{D}/{name}", "w"))
        print(f"  {name}: {dict(collections.Counter(r['dataset'] for r in obj['records']))}", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
