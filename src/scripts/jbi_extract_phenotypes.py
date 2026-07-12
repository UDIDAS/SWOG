#!/usr/bin/env python3
"""
IPKG Step 1 — extract a per-case phenotype + observability record for every
Pancreas + LiTS case, from the hand-off masks. Output is written to the JBI
folder on Desktop (persistent, survives scratch wipes).

Each record:
  case_id, dataset, observed_organs (the segmented anatomy = observability set),
  and per observed organ: {present, organ_volume_cm3, has_tumor, tumor_volume_cm3,
  tumor_diameter_mm, burden_cat, multiplicity, containment, anatomic_location}

These records are the substrate for the observability-aware retrieval pipeline.
"""
import os, json, glob
import numpy as np
import nibabel as nib
from scipy import ndimage

BUNDLE = "/scratch/ud3d4/acm_data/ssl_handoff_ours"
OUT = "/home/ud3d4/Desktop/SWOG/JBI_submission/results/case_phenotypes.json"
# dataset -> observed organ name
OBSERVED = {"pancreas": "pancreas", "lits": "liver"}


def burden_cat(vol_vox, t1, t2):
    return "low" if vol_vox < t1 else ("high" if vol_vox >= t2 else "medium")


def anat_location(seg):
    panc = np.argwhere(seg == 1); tum = np.argwhere(seg == 2)
    if len(tum) == 0 or len(panc) == 0:
        return "unknown"
    px, tx = panc[:, 0], tum[:, 0]
    ext = px.max() - px.min()
    if ext == 0:
        return "unknown"
    off = (tx.mean() - px.mean()) / ext
    return "head" if off > 0.12 else ("tail" if off < -0.12 else "body")


def extract(name, spacing_cm3):
    organ_key = OBSERVED[name]
    gt_files = sorted(glob.glob(f"{BUNDLE}/ssl_predictions/{name}/*.nii.gz"))
    # tertiles of predicted tumor volume for burden bins
    recs = []
    for f in gt_files:
        cid = os.path.basename(f).replace(".nii.gz", "")
        pr = nib.load(f).get_fdata().astype(np.uint8)
        recs.append((cid, pr, int((pr == 2).sum())))
    vols = sorted(r[2] for r in recs if r[2] > 0)
    t1, t2 = (np.percentile(vols, [33.3, 66.6]) if vols else (0, 0))

    out = []
    for cid, pr, tvox in recs:
        organ_mask = (pr == 1)
        tumor_mask = (pr == 2)
        has_tumor = bool(tumor_mask.sum() > 0)
        mult = "multifocal" if ndimage.label(tumor_mask)[1] >= 2 else "solitary"
        if has_tumor and organ_mask.sum() > 0:
            org_d = ndimage.binary_dilation(organ_mask, iterations=3)
            frac = (tumor_mask & org_d).sum() / tumor_mask.sum()
            contain = "contained" if frac >= 0.90 else "boundary"
        else:
            contain = "none"
        loc = anat_location(pr) if name == "pancreas" else "na"
        rec = {
            "case_id": cid, "dataset": name, "observed_organs": [organ_key],
            "organs": {organ_key: {
                "present": True,
                "organ_volume_cm3": round(organ_mask.sum() * spacing_cm3, 2),
                "has_tumor": has_tumor,
                "tumor_volume_cm3": round(tvox * spacing_cm3, 2),
                "tumor_voxels": tvox,
                "burden_cat": burden_cat(tvox, t1, t2) if has_tumor else "none",
                "multiplicity": mult if has_tumor else "none",
                "containment": contain,
                "anatomic_location": loc,
            }}}
        out.append(rec)
    return out


if __name__ == "__main__":
    # pancreas: native spacing varies; use per-case? For phenotype categories we
    # use voxel-count tertiles (spacing-invariant within dataset). For volume_cm3
    # display, approximate with representative spacing.
    records = []
    records += extract("pancreas", spacing_cm3=1.0 / 1000)   # voxels->cm3 approx (2D-derived)
    records += extract("lits", spacing_cm3=1.0 / 1000)       # 1mm iso -> 1mm3 -> /1000
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"n": len(records), "records": records}, open(OUT, "w"), indent=1)
    from collections import Counter
    print(f"Extracted {len(records)} case records")
    for ds in ("pancreas", "lits"):
        d = [r for r in records if r["dataset"] == ds]
        org = OBSERVED[ds]
        bc = Counter(r["organs"][org]["burden_cat"] for r in d)
        mc = Counter(r["organs"][org]["multiplicity"] for r in d)
        cc = Counter(r["organs"][org]["containment"] for r in d)
        print(f"  {ds}: n={len(d)}  burden={dict(bc)}  mult={dict(mc)}  contain={dict(cc)}")
    print(f"Saved: {OUT}")
