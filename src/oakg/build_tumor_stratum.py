"""Build the real-GT FLARE cross-organ tumor stratum (supplementary, slice-level).

Source: FLARE class-14 tumor GT merged with the 5 organ classes by slice-content
hash (produced upstream by build_flare_multiorgan.py -> flare_multiorgan_cases.json).
This is REAL ground truth (not predicted), which is why it is slice-level: the
class-stack distribution lost patient identity.

Methodological handling (the right way):
  - Evaluation-only stratum: thresholds/policies are FROZEN from the patient-level
    benchmark; nothing is fit on this data.
  - Report PAIRED method deltas, not absolute nDCG: adjacent slices of the same
    (anonymous) patient are correlated, inflating absolute scores, but every method
    faces the identical pool so paired comparisons remain valid.
  - Kept SEPARATE from the patient-level corpus (dataset name "FLARE_tumor").

Run:  PYTHONPATH=src python -m oakg.build_tumor_stratum --out data_tumor
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .build_benchmark import build_queries_relevance

DEFAULT_SRC = "/home/ud3d4/Desktop/SWOG/JBI_submission/results/flare_multiorgan_cases.json"
ORGANS = ["liver", "right_kidney", "spleen", "pancreas", "left_kidney"]


def _split_of(cid: str, ratios=(0.55, 0.20, 0.25)) -> str:
    h = int(hashlib.md5(cid.encode()).hexdigest(), 16) % 10_000 / 10_000
    return "train" if h < ratios[0] else ("val" if h < ratios[0] + ratios[1] else "test")


def build_frames(src: str, per_class_cap: int = 700, seed: int = 2027):
    rng = np.random.default_rng(seed)
    cases = json.load(open(src))["cases"]
    tumor = [c for c in cases if c["has_tumor"]]
    notum = [c for c in cases if not c["has_tumor"]]
    rng.shuffle(tumor); rng.shuffle(notum)
    sel = tumor[:per_class_cap] + notum[:per_class_cap]
    rng.shuffle(sel)

    case_rows, ref_rows = [], []
    for c in sel:
        cid, obs = c["case_id"], c["observed_organs"]
        case_rows.append({"case_id": cid, "dataset": "FLARE_tumor", "split": _split_of(cid),
                          "available_organs": "|".join(obs)})

        def add(f, t, s, v):
            ref_rows.append({"case_id": cid, "feature": f, "value": float(v),
                             "feature_type": t, "support_organs": s})

        for o in obs:
            ov = c["organs"][o]
            add(f"{o}_present", "binary", o, 1.0)
            add(f"{o}_area_px", "numeric", o, ov["organ_area_px"])
            add(f"{o}_tumor_present", "binary", o, int(ov["has_tumor"]))
            add(f"{o}_tumor_area_px", "numeric", o, ov["tumor_area_px"])
        add("has_tumor", "binary", "", int(c["has_tumor"]))
        add("cross_organ_tumor", "binary", "", int(c["cross_organ"]))
        add("multifocal", "binary", "", int(c["multiplicity"] == "multifocal"))
        add("total_tumor_area_px", "numeric", "", sum(v["tumor_area_px"] for v in c["organs"].values()))
    return pd.DataFrame(case_rows), pd.DataFrame(ref_rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the real-GT FLARE tumor stratum.")
    ap.add_argument("--src", default=DEFAULT_SRC, help="flare_multiorgan_cases.json (class-14 GT merge)")
    ap.add_argument("--out", default="data_tumor")
    ap.add_argument("--per-class-cap", type=int, default=700)
    ap.add_argument("--seed", type=int, default=2027)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cases, ref = build_frames(args.src, args.per_class_cap, args.seed)
    ref.to_csv(out / "phenotypes_ref.csv", index=False)
    ref.to_csv(out / "phenotypes_pred.csv", index=False)  # GT-only stratum: pred = ref
    cases.to_csv(out / "cases.csv", index=False)
    q, rel, thr = build_queries_relevance(cases, ref, seed=args.seed)
    q.to_csv(out / "queries.csv", index=False)
    rel.to_csv(out / "relevance.csv", index=False)

    n_tumor = int(cases["case_id"].isin(
        ref.loc[(ref.feature == "has_tumor") & (ref.value == 1), "case_id"]).sum())
    print(f"slice-cases={len(cases)} (tumor={n_tumor})  queries={len(q)}  rel_rows={len(rel)}")
    print("Wrote real-GT FLARE tumor stratum to", out.resolve(),
          "\n  NOTE: slice-level, evaluation-only; report paired deltas (absolute nDCG optimistic).")


if __name__ == "__main__":
    main()
