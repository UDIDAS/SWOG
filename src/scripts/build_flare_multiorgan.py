#!/usr/bin/env python3
"""
Reconstruct multi-organ FLARE cases from the per-class arrays.

FLARE is distributed as per-class binary arrays (one organ per file), losing the
joint multi-organ view. But the SAME physical CT slice appears in every class
whose organ it contains. We match slices across classes by CT-image content and
merge their labels -> each unique slice becomes a case that jointly observes its
actual organ set (the multi-organ coverage regime the paper needs).

Observability O = organs (of the paper's 5) present in the slice (from GT labels).
Phenotypes (from GT here; prediction-derived values plug in after training):
  per organ: has_tumor, tumor_area_px, burden_cat, containment
  case-level: cross_organ (tumor touching >=2 organs), multiplicity
Output: flare_multiorgan_cases.json + a distribution report.
"""
import os, json, hashlib
import numpy as np
from scipy import ndimage

FL = "/scratch/ud3d4/acm_data/FLARE"
OUT = "/home/ud3d4/Desktop/SWOG/JBI_submission/results/flare_multiorgan_cases.json"
ORGANS = {1: "liver", 2: "right_kidney", 3: "spleen", 4: "pancreas", 13: "left_kidney"}
TUMOR = 14


def slice_hash(img):
    return hashlib.md5(np.ascontiguousarray(img).tobytes()).hexdigest()


def build():
    # 1) hash every slice of every class; group by physical slice
    print("Hashing slices across classes...", flush=True)
    slices = {}   # hash -> {"organs": {oid: labelmask}, "tumor": mask or None, "img_ref": (class,idx)}
    for cid in list(ORGANS) + [TUMOR]:
        imgs = np.load(f"{FL}/class_{cid}_images.npy", mmap_mode="r")
        lbls = np.load(f"{FL}/class_{cid}_labels.npy", mmap_mode="r")
        N = imgs.shape[0]
        for i in range(N):
            h = slice_hash(np.asarray(imgs[i]))
            rec = slices.setdefault(h, {"organs": {}, "tumor": None})
            m = (np.asarray(lbls[i]) > 0)
            if cid == TUMOR:
                rec["tumor"] = m
            else:
                rec["organs"][cid] = m
        print(f"  class {cid} ({ORGANS.get(cid,'tumor')}): {N} slices, total unique so far {len(slices)}", flush=True)

    # 2) build per-slice case records
    print("Building case records...", flush=True)
    cases = []
    for h, rec in slices.items():
        observed = sorted(rec["organs"].keys())
        if not observed:
            continue  # tumor-only slice with none of our 5 organs -> skip (organ outside scope)
        tumor = rec["tumor"]
        organs_out = {}
        cross = 0
        # FLARE23 tumors are peri-organ (abut the tight organ boundary); assign with a 6px margin
        MARGIN = 6
        for oid in observed:
            om = rec["organs"][oid]
            area = int(om.sum())
            om_d = ndimage.binary_dilation(om, iterations=MARGIN) if tumor is not None else om
            touch = int((tumor & om_d).sum()) if tumor is not None else 0
            has_t = touch > 0
            if has_t:
                tarea = touch
                inside = (tumor & om).sum() / max(1, tumor.sum())   # exact-overlap fraction
                contain = "contained" if inside >= 0.5 else "boundary"
                burden = "high" if tumor.sum() > 400 else ("low" if tumor.sum() < 80 else "medium")
                cross += 1
            else:
                tarea, contain, burden = 0, "none", "none"
            organs_out[ORGANS[oid]] = {"organ_area_px": area, "has_tumor": has_t,
                                       "tumor_area_px": tarea, "burden_cat": burden, "containment": contain}
        has_tumor_any = tumor is not None and tumor.sum() > 0
        mult = "none"
        if has_tumor_any:
            mult = "multifocal" if ndimage.label(tumor)[1] >= 2 else "solitary"
        cases.append({
            "case_id": f"FLARE-{h[:10]}", "dataset": "flare", "granularity": "slice",
            "observed_organs": [ORGANS[o] for o in observed],
            "n_observed": len(observed),
            "organs": organs_out,
            "has_tumor": bool(has_tumor_any),
            "tumor_in_observed_organs": cross,     # #observed organs containing tumor
            "cross_organ": cross >= 2,             # tumor touches >=2 observed organs
            "tumor_in_unobserved": bool(has_tumor_any and cross == 0),  # tumor present but in none of our 5
            "multiplicity": mult,
        })

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"n": len(cases), "organs": list(ORGANS.values()), "cases": cases}, open(OUT, "w"))

    # 3) distribution report
    from collections import Counter
    nobs = Counter(c["n_observed"] for c in cases)
    print(f"\n=== FLARE multi-organ reconstruction: {len(cases)} slice-cases ===")
    print("Observed-organ-count distribution:")
    for k in sorted(nobs):
        print(f"  observes {k} organ(s): {nobs[k]} cases")
    multi = sum(1 for c in cases if c["n_observed"] >= 2)
    print(f"MULTI-ORGAN cases (>=2 organs): {multi}  ({100*multi/len(cases):.0f}%)")
    print(f"cases with tumor: {sum(c['has_tumor'] for c in cases)}")
    print(f"cross-organ tumor cases (tumor in >=2 organs): {sum(c['cross_organ'] for c in cases)}")
    print(f"tumor-in-unobserved cases: {sum(c['tumor_in_unobserved'] for c in cases)}")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    build()
