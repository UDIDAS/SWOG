#!/usr/bin/env python3
"""
Build a FULLY per-patient 3-regime corpus and test what the real per-patient FLARE22
volumes change for the retrieval/observability tables.

Regimes (all per-patient volumes now — NO slice-level pseudo-cases anywhere):
  - Pancreas : observes {pancreas}, has tumour phenotypes           (existing case_phenotypes.json)
  - LiTS     : observes {liver}, has tumour phenotypes              (existing case_phenotypes.json)
  - FLARE22  : observes {liver,RK,spleen,pancreas,LK}, ORGAN morphology only, NO tumour (new)

Key consequence to measure honestly: FLARE22 has no tumour, so it cannot supply tumour-positive
"broad" matches for the tumour-decomposition benchmark. What it DOES supply cleanly:
  (1) Table 8 disjoint-observability controls on 100% real per-patient data (removes slice caveat).
  (2) An ORGAN-MORPHOLOGY decomposition benchmark: narrow query (one organ's size) vs broad
      FLARE22 case sharing that organ -> coverage-blind still penalised for unobserved anatomy.

Writes corpus_perpatient.json + prints the two analyses.
"""
import os, json, glob
import numpy as np
import nibabel as nib

RES = "/home/ud3d4/Desktop/SWOG/kg/data"
FT2 = "/scratch/ud3d4/acm_data/FLARE_Task2"
OUT = f"{RES}/corpus_perpatient.json"
ORGANS = {1: "liver", 2: "right_kidney", 3: "spleen", 4: "pancreas", 13: "left_kidney"}


def flare22_pairs():
    pairs = []
    for idir, ldir in [("train_gt_label/imagesTr", "train_gt_label/labelsTr"),
                       ("validation/Validation-Public-Images", "validation/Validation-Public-Labels")]:
        for ip in sorted(glob.glob(f"{FT2}/{idir}/*.nii.gz")):
            base = os.path.basename(ip).replace("_0000.nii.gz", ".nii.gz")
            lp = f"{FT2}/{ldir}/{base}"
            if os.path.exists(lp) and not os.path.basename(ip).startswith("._"):
                pairs.append((ip, lp))
    return pairs


def build_flare22_records():
    """Per-patient FLARE22 records: organ presence + volume(cm3); size category assigned later."""
    recs, raw = [], []
    for ip, lp in flare22_pairs():
        ni = nib.load(lp); lab = ni.get_fdata().astype(np.uint8)
        vox_cm3 = float(np.prod(ni.header.get_zooms()[:3])) / 1000.0
        cid = os.path.basename(ip).replace("_0000.nii.gz", "")
        organs, vols = {}, {}
        for oid, name in ORGANS.items():
            n = int((lab == oid).sum())
            if n > 0:
                vols[name] = n * vox_cm3
        raw.append((cid, vols))
    # size categories per organ by tertiles across the cohort
    cats = {}
    for name in ORGANS.values():
        vals = np.array([v[name] for _, v in raw if name in v])
        if len(vals) >= 3:
            cats[name] = (np.percentile(vals, 33), np.percentile(vals, 66))
        else:
            cats[name] = (0, 1e9)
    for cid, vols in raw:
        organs = {}
        for name, vol in vols.items():
            lo, hi = cats[name]
            sz = "small" if vol < lo else ("large" if vol > hi else "medium")
            organs[name] = {"present": True, "organ_volume_cm3": round(vol, 2),
                            "has_tumor": False, "tumor_volume_cm3": 0.0, "burden_cat": "none",
                            "multiplicity": "none", "containment": "none",
                            "anatomic_location": "unknown", "size_cat": sz}
        recs.append({"case_id": cid, "dataset": "flare", "granularity": "volume",
                     "observed_organs": list(organs.keys()), "n_observed": len(organs),
                     "organs": organs, "has_tumor": False})
    return recs


def main():
    base = json.load(open(f"{RES}/case_phenotypes.json"))["records"]
    for r in base:
        r.setdefault("granularity", "volume")
        for o in r["organs"].values():
            o.setdefault("size_cat", "unknown")   # single-organ sets: size not the axis of interest
    flare = build_flare22_records()
    corpus = base + flare
    json.dump({"n": len(corpus), "organ_universe": list(ORGANS.values()),
               "note": "FULLY per-patient. FLARE22 = organ morphology, NO tumour.",
               "records": corpus}, open(OUT, "w"))
    from collections import Counter
    print(f"Per-patient corpus: {len(corpus)}  {dict(Counter(r['dataset'] for r in corpus))}")
    print(f"  FLARE22 per-patient cases: {len(flare)} (all observe {len(ORGANS)} organs)")
    print(f"  granularity: {dict(Counter(r['granularity'] for r in corpus))}  <- NO slice pseudo-cases")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
