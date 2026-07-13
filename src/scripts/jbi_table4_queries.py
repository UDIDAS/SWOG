#!/usr/bin/env python3
"""
JBI Table 4 — Structured query reasoning with three-valued (T/F/Indeterminate) semantics.

For each query pattern we evaluate every case under strong-Kleene semantics:
  True/False  when the case OBSERVED the anatomy the query references (resolved),
  Indeterminate when the query references anatomy the case did NOT observe.
Precision/Recall/F1 are computed over the resolved (T/F) verdicts (system=extracted
phenotype vs reference=GT phenotype); Indet. = fraction indeterminate (reported
separately). This is the key non-collapse property: a coverage-blind system would
return False for indeterminate cases, manufacturing false negatives.

Datasets: Pancreas (observes pancreas) + LiTS (observes liver).
"""
import os, glob, json
import numpy as np
import nibabel as nib
from scipy import ndimage

BUNDLE = "/scratch/ud3d4/acm_data/ssl_handoff_ours"
OUT = "/home/ud3d4/Desktop/SWOG/JBI_submission/results"
OBSERVED = {"pancreas": "pancreas", "lits": "liver"}


def phen_from_mask(seg, t1, t2):
    tvox = int((seg == 2).sum())
    has = tvox > 0
    burden = "none" if not has else ("low" if tvox < t1 else ("high" if tvox >= t2 else "medium"))
    mult = "none" if not has else ("multifocal" if ndimage.label(seg == 2)[1] >= 2 else "solitary")
    return {"has_tumor": has, "burden": burden, "multiplicity": mult, "tvox": tvox}


def load(name):
    organ = OBSERVED[name]
    files = sorted(glob.glob(f"{BUNDLE}/ssl_predictions/{name}/*.nii.gz"))
    recs = []
    for f in files:
        cid = os.path.basename(f).replace(".nii.gz", "")
        pr = nib.load(f).get_fdata().astype(np.uint8)
        gt = nib.load(f"{BUNDLE}/ground_truth/{name}/{cid}.nii.gz").get_fdata().astype(np.uint8)
        recs.append((cid, organ, pr, gt))
    # tertiles per dataset from GT tumor voxels
    gv = sorted(int((gt == 2).sum()) for _, _, _, gt in recs if (gt == 2).any())
    t1, t2 = (np.percentile(gv, [33.3, 66.6]) if gv else (0, 0))
    out = []
    for cid, organ, pr, gt in recs:
        out.append({"case_id": cid, "observed": organ,
                    "pred": phen_from_mask(pr, t1, t2), "gt": phen_from_mask(gt, t1, t2)})
    return out


def prf(sys_true, ref_true):
    tp = sum(1 for s, r in zip(sys_true, ref_true) if s and r)
    fp = sum(1 for s, r in zip(sys_true, ref_true) if s and not r)
    fn = sum(1 for s, r in zip(sys_true, ref_true) if not s and r)
    P = tp / (tp + fp) if tp + fp else float("nan")
    R = tp / (tp + fn) if tp + fn else float("nan")
    F = 2 * P * R / (P + R) if (P == P and R == R and P + R) else float("nan")
    return round(P, 3), round(R, 3), round(F, 3)


def query(cases, predicate, references_organ=None):
    """Three-valued: predicate(rec,'pred') is the system verdict, ('gt') the reference.
    Indeterminate if references_organ is set and the case didn't observe it."""
    sys_t, ref_t = [], []
    indet = 0
    for c in cases:
        if references_organ is not None and c["observed"] != references_organ:
            indet += 1
            continue
        sys_t.append(predicate(c, "pred"))
        ref_t.append(predicate(c, "gt"))
    n_res = len(sys_t)
    P, R, F = prf(sys_t, ref_t) if n_res else (float("nan"),) * 3
    return {"n_q": len(cases), "resolved": n_res, "precision": P, "recall": R, "f1": F,
            "indet_pct": round(100 * indet / len(cases), 1)}


if __name__ == "__main__":
    cases = load("pancreas") + load("lits")
    print(f"Cases: {len(cases)}")

    rows = {}
    # Q1 High tumor burden — determinate (observed organ's burden always known)
    rows["High tumor burden"] = query(cases, lambda c, k: c[k]["burden"] == "high")
    # Q2 Multifocal disease — determinate
    rows["Multifocal disease"] = query(cases, lambda c, k: c[k]["multiplicity"] == "multifocal")
    # Q3 Tumor in specified organ — average of "in pancreas" and "in liver" (indeterminate elsewhere)
    q_panc = query(cases, lambda c, k: c[k]["has_tumor"], references_organ="pancreas")
    q_liv = query(cases, lambda c, k: c[k]["has_tumor"], references_organ="liver")
    rows["Tumor in specified organ"] = {
        "n_q": len(cases), "resolved": q_panc["resolved"] + q_liv["resolved"],
        "precision": round(np.nanmean([q_panc["precision"], q_liv["precision"]]), 3),
        "recall": round(np.nanmean([q_panc["recall"], q_liv["recall"]]), 3),
        "f1": round(np.nanmean([q_panc["f1"], q_liv["f1"]]), 3),
        "indet_pct": round((q_panc["indet_pct"] + q_liv["indet_pct"]) / 2, 1),
        "per_organ": {"pancreas": q_panc, "liver": q_liv}}
    # Q4 Cross-organ distribution — needs >=2 observed organs; single-organ cases => indeterminate
    rows["Cross-organ distribution"] = {
        "n_q": len(cases), "resolved": 0, "precision": None, "recall": None, "f1": None,
        "indet_pct": 100.0,
        "note": "All cases observe a single organ; cross-organ presence is unobservable -> indeterminate "
                "(strong-Kleene prevents false negatives). Requires multi-organ per-patient cases to resolve."}

    # Mean over resolvable rows
    resolvable = [r for r in ["High tumor burden", "Multifocal disease", "Tumor in specified organ"]]
    mean = {"precision": round(np.nanmean([rows[r]["precision"] for r in resolvable]), 3),
            "recall": round(np.nanmean([rows[r]["recall"] for r in resolvable]), 3),
            "f1": round(np.nanmean([rows[r]["f1"] for r in resolvable]), 3),
            "indet_pct": round(np.mean([rows[r]["indet_pct"] for r in resolvable + ["Cross-organ distribution"]]), 1)}
    rows["Mean"] = {"n_q": len(cases), **mean}

    print("\n=== JBI Table 4: Structured query reasoning (three-valued) ===")
    print(f"{'Query pattern':<26s} {'n_q':>5s} {'P':>6s} {'R':>6s} {'F1':>6s} {'Indet%':>7s}")
    for name, r in rows.items():
        p = "-" if r.get("precision") is None else f"{r['precision']}"
        rr = "-" if r.get("recall") is None else f"{r['recall']}"
        f = "-" if r.get("f1") is None else f"{r['f1']}"
        print(f"{name:<26s} {r['n_q']:>5d} {p:>6s} {rr:>6s} {f:>6s} {r['indet_pct']:>7.1f}")

    json.dump({"table": "Table 4 - Structured query reasoning (three-valued semantics)",
               "caption_source": "JBI_VKG_2026.pdf Table 4",
               "method": "Strong-Kleene T/F/Indeterminate per case; P/R/F1 over resolved (T/F) verdicts "
                         "(system=SAM3-extracted phenotype vs reference=GT); Indet=fraction referencing "
                         "unobserved anatomy. Pancreas+LiTS (single-organ observability).",
               "rows": rows}, open(f"{OUT}/table4_structured_queries.json", "w"), indent=2)
    print(f"\nSaved: {OUT}/table4_structured_queries.json")
