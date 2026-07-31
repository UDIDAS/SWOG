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
import zlib

import nibabel as nib
import numpy as np
from scipy import ndimage

OUT = "/scratch/ud3d4/acm_data/flare23_labels"
DEST = "/home/ud3d4/Desktop/SWOG/kg/data/corpus_flare_train.json"
LABEL_ORGAN = {1: "liver", 2: "right_kidney", 3: "spleen", 4: "pancreas", 13: "left_kidney"}
TMP = "/dev/shm/_flare_mask.nii.gz"


def pheno(mask, sp):
    organs, observed = {}, []
    tm = mask == 14
    tvox = int(tm.sum())
    torgan, best = None, -1
    if tvox:
        for lab, o in LABEL_ORGAN.items():
            ov = int((tm & (mask == lab)).sum())
            if ov > best:
                best, torgan = ov, o
        if best <= 0:
            torgan = None
    for lab, o in LABEL_ORGAN.items():
        om = mask == lab
        ov = int(om.sum())
        if ov < 20:
            continue
        observed.append(o)
        ht = o == torgan and tvox > 0
        mult, contain, loc = "none", "none", "na"
        if ht:
            xs, ys, zs = np.where(tm)
            sub = tm[xs.min():xs.max() + 1, ys.min():ys.max() + 1, zs.min():zs.max() + 1]
            mult = "multifocal" if ndimage.label(sub)[1] >= 2 else "solitary"
            contain = "contained" if int((tm & om).sum()) / tvox >= 0.5 else "boundary"
            if o == "pancreas":
                loc = "unknown"
        organs[o] = {"present": True, "organ_volume_cm3": round(ov * sp, 2), "has_tumor": ht,
                     "tumor_volume_cm3": round(tvox * sp, 2) if ht else 0.0,
                     "tumor_voxels": tvox if ht else 0, "burden_cat": "pending" if ht else "none",
                     "multiplicity": mult, "containment": contain, "anatomic_location": loc,
                     "size_cat": "unknown"}
    return {"dataset": "flare", "observed_organs": observed, "n_observed": len(observed),
            "organs": organs, "granularity": "volume"}


def main():
    idx = json.load(open(f"{OUT}/label_index.json"))
    first, labs = idx["first_offset"], idx["labels"]
    buf = open(f"{OUT}/flare_labels.bin", "rb").read()
    records = []
    for n, (name, off, csize) in enumerate(labs):
        p = off - first
        if buf[p:p + 4] != b"PK\x03\x04":
            continue
        nlen = struct.unpack("<H", buf[p + 26:p + 28])[0]
        elen = struct.unpack("<H", buf[p + 28:p + 30])[0]
        ds = p + 30 + nlen + elen
        try:
            raw = zlib.decompress(buf[ds:ds + csize], -15)
            open(TMP, "wb").write(raw)
            img = nib.load(TMP)
            mask = np.asarray(img.dataobj).astype(np.uint8)
            sp = float(np.prod(img.header.get_zooms()[:3])) / 1000.0
        except Exception as e:
            print(f"  skip {name}: {str(e)[:60]}", flush=True)
            continue
        rec = pheno(mask, sp)
        if rec["observed_organs"]:
            rec["case_id"] = os.path.basename(name).replace(".nii.gz", "")
            records.append(rec)
        if (n + 1) % 100 == 0:
            print(f"  {n+1}/{len(labs)} processed, {len(records)} kept", flush=True)

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
               "organ+tumor, LiTS/Pancreas schema"}, open(DEST, "w"))
    print("DONE:", json.dumps(summary), flush=True)
    print("->", DEST, flush=True)


if __name__ == "__main__":
    main()
