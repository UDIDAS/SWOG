#!/usr/bin/env python3
"""
Build the FLARE23 TRAIN KG (organ + tumor, patient-level) from the label masks extracted from
Metadata.zip's contiguous label byte-range. Matches the LiTS/Pancreas corpus schema so FLARE is
finally comparable (organs AND tumor). GT-derived (train reference).

in:  /scratch/ud3d4/acm_data/flare23_labels/{flare_labels.bin, label_index.json}
out: kg/data/corpus_flare_train.json  (records[] + summary)
"""
import json
import os
import struct
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor

import nibabel as nib
import numpy as np
from scipy import ndimage

N_WORKERS = int(os.environ.get("KG_WORKERS", "8"))

OUT = "/scratch/ud3d4/acm_data/flare23_labels"
DEST = "/home/ud3d4/Desktop/SWOG/kg/data/corpus_flare_train.json"
# all 13 FLARE23 organ labels (14 = tumor) — richer than the 5-organ LiTS/Pancreas set
LABEL_ORGAN = {1: "liver", 2: "right_kidney", 3: "spleen", 4: "pancreas", 5: "aorta",
               6: "inferior_vena_cava", 7: "right_adrenal", 8: "left_adrenal", 9: "gallbladder",
               10: "esophagus", 11: "stomach", 12: "duodenum", 13: "left_kidney"}
TMP = "/dev/shm/_flare_mask.nii.gz"


def _feret_mm(mask, zooms):
    """Max caliper (Feret) diameter in mm — convex hull of the (sub-sampled) voxels."""
    coords = np.argwhere(mask)
    if len(coords) < 2:
        return 0.0
    if len(coords) > 4000:                       # bound cost on big organs; keep axis extremes
        keep = set(coords[coords[:, ax].argmin()].tobytes() for ax in range(3))
        keep |= set(coords[coords[:, ax].argmax()].tobytes() for ax in range(3))
        idx = np.linspace(0, len(coords) - 1, 4000).astype(int)
        coords = coords[idx]
    pts = coords * np.asarray(zooms, float)
    try:
        from scipy.spatial import ConvexHull
        pts = pts[ConvexHull(pts).vertices]
    except Exception:
        pass
    from scipy.spatial.distance import pdist
    return round(float(pdist(pts).max()), 1)


def _centroid_mm(mask, affine):
    c = ndimage.center_of_mass(mask)
    w = nib.affines.apply_affine(affine, np.array(c, float))
    return [round(float(x), 1) for x in w]


def pheno(mask, img):
    zooms = img.header.get_zooms()[:3]
    sp = float(np.prod(zooms)) / 1000.0
    affine = img.affine
    organs, observed = {}, []
    tm = mask == 14
    tvox = int(tm.sum())
    torgan, best = None, 0
    tdia, tcen, lesion_count = 0.0, None, 0
    if tvox:
        # FLARE23 tumor (label 14) is disjoint from organ voxels; dilate to find its host organ.
        tmd = ndimage.binary_dilation(tm, iterations=3)
        for lab, o in LABEL_ORGAN.items():
            ov = int((tmd & (mask == lab)).sum())
            if ov > best:
                best, torgan = ov, o
        tdia = _feret_mm(tm, zooms)
        tcen = _centroid_mm(tm, affine)
        lesion_count = int(ndimage.label(tm)[1])
    for lab, o in LABEL_ORGAN.items():
        om = mask == lab
        ov = int(om.sum())
        if ov < 20:
            continue
        observed.append(o)
        ht = o == torgan and tvox > 0
        mult, loc = "none", "na"
        if ht:
            mult = "multifocal" if lesion_count >= 2 else "solitary"
            if o == "pancreas":
                loc = "unknown"
        organs[o] = {"present": True,
                     "organ_volume_cm3": round(ov * sp, 2), "organ_voxels": ov,
                     "organ_max_diameter_mm": _feret_mm(om, zooms),
                     "organ_centroid_mm": _centroid_mm(om, affine),
                     "has_tumor": ht,
                     "tumor_volume_cm3": round(tvox * sp, 2) if ht else 0.0,
                     "tumor_voxels": tvox if ht else 0,
                     "tumor_max_diameter_mm": tdia if ht else 0.0,
                     "tumor_centroid_mm": tcen if ht else None,
                     "lesion_count": lesion_count if ht else 0,
                     "burden_cat": "pending" if ht else "none",
                     "multiplicity": mult, "containment": "na", "anatomic_location": loc,
                     "size_cat": "unknown"}
    return {"dataset": "flare", "observed_organs": observed, "n_observed": len(observed),
            "organs": organs, "granularity": "volume"}


def main():
    idx = json.load(open(f"{OUT}/label_index.json"))
    first, labs = idx["first_offset"], idx["labels"]
    buf = open(f"{OUT}/flare_labels.bin", "rb").read()

    def worker(entry):
        name, off, csize = entry
        p = off - first
        if buf[p:p + 4] != b"PK\x03\x04":
            return None
        nlen = struct.unpack("<H", buf[p + 26:p + 28])[0]
        elen = struct.unpack("<H", buf[p + 28:p + 30])[0]
        ds = p + 30 + nlen + elen
        tmp = f"/dev/shm/_flare_{threading.get_ident()}.nii.gz"   # per-thread temp (no collision)
        try:
            open(tmp, "wb").write(zlib.decompress(buf[ds:ds + csize], -15))
            img = nib.load(tmp)
            mask = np.asarray(img.dataobj).astype(np.uint8)
        except Exception:
            return None
        rec = pheno(mask, img)
        if rec["observed_organs"]:
            rec["case_id"] = os.path.basename(name).replace(".nii.gz", "")
            return rec
        return None

    records = []
    print(f"processing {len(labs)} masks on {N_WORKERS} threads...", flush=True)
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        for i, rec in enumerate(ex.map(worker, labs)):
            if rec:
                records.append(rec)
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(labs)} processed, {len(records)} kept", flush=True)

    # burden tertiles over observed tumor volumes
    tvols = sorted(od["tumor_volume_cm3"] for r in records for od in r["organs"].values()
                   if od["has_tumor"])
    t1, t2 = (np.percentile(tvols, [33.3, 66.6]) if tvols else (0, 0))
    for r in records:
        for od in r["organs"].values():
            if od["burden_cat"] == "pending":
                od["burden_cat"] = "high" if od["tumor_volume_cm3"] >= t2 else \
                    ("low" if od["tumor_volume_cm3"] <= t1 else "medium")
    n_tumor = sum(1 for r in records if any(od["has_tumor"] for od in r["organs"].values()))
    from collections import Counter
    org_cov = Counter(o for r in records for o in r["observed_organs"])
    summary = {"n_records": len(records), "with_tumor": n_tumor,
               "organ_coverage": dict(org_cov), "burden_tertiles_cm3": [round(t1, 2), round(t2, 2)]}
    json.dump({"records": records, "summary": summary, "note": "FLARE23 train KG, GT-derived, "
               "13 organs + tumor, enriched: volume_cm3/voxels/max_diameter_mm/centroid_mm/lesion_count"},
              open(DEST, "w"))
    print("DONE:", json.dumps(summary), flush=True)
    print("->", DEST, flush=True)


if __name__ == "__main__":
    main()
