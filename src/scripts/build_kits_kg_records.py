#!/usr/bin/env python3
"""Add the KiTS23 train set to the knowledge graph — labels-only path (like FLARE23): download each
small segmentation.nii.gz (kidney=1, tumor=2), compute per-patient kidney + tumor phenotypes, discard the
mask. No images needed, storage-safe, thread-pooled. Merges KiTS records into the unified KG corpora
(corpus_perpatient.json + corpus_global.json) and writes corpus_kits.json. Burden matches the existing
cohort's tertile thresholds (~5 cc / ~20 cc).
"""
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

D = "/home/ud3d4/Desktop/SWOG/kg/data"
SEG_URL = "https://raw.githubusercontent.com/neheller/kits23/main/dataset/case_{:05d}/segmentation.nii.gz"
N_CASES, WORKERS = 489, 8          # KiTS23 public train set = case_00000 .. case_00488
KIDNEY, TUMOR = 1, 2               # KiTS label scheme (3 = cyst, ignored here)


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


def burden(cc):
    return "none" if cc <= 0 else "low" if cc < 5 else "medium" if cc < 20 else "high"


def one_case(cn):
    seg_p = f"/dev/shm/_kits_kg_{cn}.nii.gz"
    try:
        urllib.request.urlretrieve(SEG_URL.format(cn), seg_p)
        nii = nib.load(seg_p); seg = np.asarray(nii.dataobj).astype(np.uint8)
        sp = [float(z) for z in nii.header.get_zooms()[:3]]
        vox = float(np.prod(sp)) / 1000.0                                   # cm^3 per voxel
        kid = seg == KIDNEY; tum = seg == TUMOR
        if int(kid.sum()) < 50:
            return None
        lbl, n_les = ndimage.label(tum)
        tcc = int(tum.sum()) * vox
        rec = {"dataset": "kits", "case_id": f"case_{cn:05d}", "granularity": "volume",
               "observed_organs": ["kidney"], "n_observed": 1,
               "organs": {"kidney": {
                   "present": True,
                   "organ_volume_cm3": round(int(kid.sum()) * vox, 2),
                   "organ_voxels": int(kid.sum()),
                   "organ_max_diameter_mm": feret_mm(kid, sp),
                   "organ_centroid_mm": [round(float(c), 1) for c in (np.argwhere(kid).mean(0) * sp)],
                   "has_tumor": bool(tum.sum() >= 50),
                   "tumor_volume_cm3": round(tcc, 2),
                   "tumor_voxels": int(tum.sum()),
                   "tumor_max_diameter_mm": feret_mm(tum, sp) if tum.sum() >= 50 else 0.0,
                   "tumor_centroid_mm": ([round(float(c), 1) for c in (np.argwhere(tum).mean(0) * sp)]
                                         if tum.sum() >= 50 else None),
                   "lesion_count": int(n_les),
                   "burden_cat": burden(tcc if tum.sum() >= 50 else 0),
                   "multiplicity": "none" if tum.sum() < 50 else "solitary" if n_les == 1 else "multifocal",
                   "containment": "contained" if tum.sum() >= 50 else "none",
                   "anatomic_location": "na", "size_cat": "unknown"}}}
        return rec
    except urllib.error.HTTPError:
        return None
    except Exception as e:
        return {"_err": f"case_{cn:05d}: {type(e).__name__} {str(e)[:60]}"}
    finally:
        if os.path.exists(seg_p):
            os.remove(seg_p)


def main():
    print(f"downloading + phenotyping KiTS train ({N_CASES} cases, {WORKERS} workers)...", flush=True)
    with ThreadPoolExecutor(WORKERS) as ex:
        out = list(ex.map(one_case, range(N_CASES)))
    recs = [r for r in out if r and "_err" not in r]
    errs = [r["_err"] for r in out if r and "_err" in r]
    wt = sum(1 for r in recs if r["organs"]["kidney"]["has_tumor"])
    print(f"KiTS KG records: {len(recs)} kidney patients ({wt} with tumor); errors: {len(errs)}", flush=True)
    json.dump({"records": recs}, open(f"{D}/corpus_kits.json", "w"))

    # merge into the unified KG corpora (replace any prior kits, keep others)
    for name in ["corpus_perpatient.json", "corpus_global.json"]:
        obj = json.load(open(f"{D}/{name}"))
        keep = [r for r in obj["records"] if r.get("dataset") != "kits"]
        obj["records"] = keep + recs
        json.dump(obj, open(f"{D}/{name}", "w"))
        import collections
        print(f"  {name}: {dict(collections.Counter(r['dataset'] for r in obj['records']))}", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
