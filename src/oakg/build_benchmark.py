"""Build the five OAKG benchmark CSVs from paired GT/predicted segmentation masks.

Reference (GT) masks populate ``phenotypes_ref.csv`` and define relevance;
predicted (SSL/SAM3) masks populate ``phenotypes_pred.csv`` (end-to-end track).
Each source dataset covers one organ, so per-source organ coverage is the
heterogeneous observability structure OAKG operates on.

Run:  PYTHONPATH=src python -m oakg.build_benchmark --out data [--limit N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .phenotypes import DatasetSpec, extract_phenotypes

# --- Source registry -------------------------------------------------------
ACM = Path("/scratch/ud3d4/acm_data/ssl_handoff_ours")


def default_sources() -> list[dict]:
    return [
        {
            "spec": DatasetSpec("Pancreas", "pancreas", organ_label=1, tumor_label=2),
            "gt_dir": ACM / "ground_truth" / "pancreas",
            "pred_dir": ACM / "ssl_predictions" / "pancreas",
        },
        {
            "spec": DatasetSpec("LiTS", "liver", organ_label=1, tumor_label=2),
            "gt_dir": ACM / "ground_truth" / "lits",
            "pred_dir": ACM / "ssl_predictions" / "lits",
        },
    ]


# Feature registry: how a phenotype dict maps to (feature, feature_type, support).
# ``support=""`` -> globally observed; else supported by the named organ.
def _feature_rows(pheno: dict, organ: str) -> list[tuple[str, str, str, float]]:
    rows = [
        ("has_tumor", "binary", "", pheno["tumor_present"]),
        ("tumor_burden_cm3", "numeric", "", pheno["tumor_burden_cm3"]),
        ("lesion_multiplicity", "numeric", "", pheno["lesion_multiplicity"]),
        (f"{organ}_present", "binary", organ, pheno["organ_present"]),
        (f"{organ}_tumor_present", "binary", organ, pheno["tumor_present"]),
        (f"{organ}_tumor_containment", "binary", organ, pheno["tumor_containment"]),
        (f"{organ}_volume_cm3", "numeric", organ, pheno["organ_volume_cm3"]),
    ]
    # Drop observations the mask could not define (None -> missing downstream).
    return [(f, t, s, float(v)) for (f, t, s, v) in rows if v is not None]


def _assign_split(case_id: str, ratios=(0.55, 0.20, 0.25)) -> str:
    """Deterministic patient-level split from a stable hash of the case id."""
    h = int(hashlib.md5(case_id.encode()).hexdigest(), 16) % 10_000 / 10_000
    if h < ratios[0]:
        return "train"
    if h < ratios[0] + ratios[1]:
        return "val"
    return "test"


def _read_mask(path: Path):
    import nibabel as nib
    img = nib.load(str(path))
    arr = np.asanyarray(img.dataobj)
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    return arr, spacing


def build_phenotype_frames(sources: list[dict], limit: int | None = None):
    case_rows, ref_rows, pred_rows = [], [], []
    for src in sources:
        spec: DatasetSpec = src["spec"]
        gt_files = sorted(Path(src["gt_dir"]).glob("*.nii.gz"))
        if limit:
            gt_files = gt_files[:limit]
        for gt_path in gt_files:
            case_id = gt_path.name.replace(".nii.gz", "")
            pred_path = Path(src["pred_dir"]) / gt_path.name

            gt_arr, gt_sp = _read_mask(gt_path)
            ref_ph = extract_phenotypes(gt_arr, gt_sp, spec)
            for f, t, s, v in _feature_rows(ref_ph, spec.organ):
                ref_rows.append({"case_id": case_id, "feature": f, "value": v,
                                 "feature_type": t, "support_organs": s})

            if pred_path.exists():
                pr_arr, pr_sp = _read_mask(pred_path)
                pred_ph = extract_phenotypes(pr_arr, pr_sp, spec)
                for f, t, s, v in _feature_rows(pred_ph, spec.organ):
                    pred_rows.append({"case_id": case_id, "feature": f, "value": v,
                                      "feature_type": t, "support_organs": s})

            case_rows.append({
                "case_id": case_id,
                "dataset": spec.name,
                "split": _assign_split(case_id),
                "available_organs": spec.organ,
            })
    return (pd.DataFrame(case_rows),
            pd.DataFrame(ref_rows),
            pd.DataFrame(pred_rows))


# --- Queries and relevance (reference-defined) -----------------------------
QUERYABLE_BINARY = ["has_tumor"]         # organ tumor-present features added per case
NUMERIC_THRESHOLDS = {"lesion_multiplicity": 2.0}  # burden threshold fit below


def build_queries_relevance(
    cases: pd.DataFrame,
    ref: pd.DataFrame,
    seed: int = 2027,
):
    rng = np.random.default_rng(seed)
    # Wide reference table: case_id x feature -> value.
    wide = ref.pivot_table(index="case_id", columns="feature", values="value")
    organ_of = dict(zip(cases["case_id"], cases["available_organs"]))
    support_of = dict(zip(ref["feature"], ref["support_organs"]))

    # Burden threshold fit on train+val only (leakage control).
    fit_ids = cases.loc[cases["split"].isin(["train", "val"]), "case_id"]
    burden = wide.reindex(fit_ids)["tumor_burden_cm3"].dropna()
    burden_thr = float(np.quantile(burden, 0.55)) if len(burden) else 0.0
    thresholds = {**NUMERIC_THRESHOLDS, "tumor_burden_cm3": burden_thr}

    all_ids = cases["case_id"].tolist()
    test_ids = cases.loc[cases["split"].eq("test"), "case_id"].tolist()

    def support_list(feat: str) -> list[str]:
        s = support_of.get(feat, "")
        return [s] if s else []

    def make_predicate(feat: str, case_id: str) -> dict:
        if feat in thresholds:
            return {"feature": feat, "op": ">=", "value": thresholds[feat],
                    "support_organs": support_list(feat)}
        return {"feature": feat, "op": "==", "value": int(wide.at[case_id, feat]),
                "support_organs": support_list(feat)}

    def holds(pred: dict, cand_id: str) -> bool:
        feat = pred["feature"]
        if feat not in wide.columns:
            return False
        v = wide.at[cand_id, feat]
        if pd.isna(v):
            return False
        if pred["op"] == ">=":
            return v >= pred["value"]
        return v == pred["value"]

    query_rows, rel_rows = [], []
    qid = 0
    for qcase in test_ids:
        organ = organ_of[qcase]
        # Candidate query features observed for this case.
        pool = ["has_tumor", "tumor_burden_cm3", "lesion_multiplicity", f"{organ}_tumor_present"]
        pool = [f for f in pool if f in wide.columns and not pd.isna(wide.at[qcase, f])]
        if len(pool) < 1:
            continue
        n_pick = min(2, len(pool))
        chosen = rng.choice(pool, size=n_pick, replace=False).tolist()
        primary = [make_predicate(chosen[0], qcase)]
        secondary = [make_predicate(f, qcase) for f in chosen[1:]]

        query_id = f"Q{qid:04d}"; qid += 1
        query_rows.append({
            "query_id": query_id, "query_case_id": qcase,
            "primary_predicates": json.dumps(primary),
            "secondary_predicates": json.dumps(secondary),
        })
        for cand in all_ids:
            if cand == qcase:
                continue
            p_ok = all(holds(p, cand) for p in primary)
            s_count = sum(holds(p, cand) for p in secondary)
            binary = int(p_ok and s_count == len(secondary))
            if not p_ok:
                grade = 0.0
            elif not secondary:
                grade = 3.0
            else:
                grade = 1.0 + 2.0 * s_count / len(secondary)
            rel_rows.append({"query_id": query_id, "candidate_id": cand,
                             "binary_relevance": binary, "graded_relevance": grade})

    return pd.DataFrame(query_rows), pd.DataFrame(rel_rows), thresholds


def main() -> None:
    ap = argparse.ArgumentParser(description="Build OAKG benchmark CSVs from masks.")
    ap.add_argument("--out", default="data", help="output directory for the 5 CSVs")
    ap.add_argument("--limit", type=int, default=None, help="cap cases per dataset (debug)")
    ap.add_argument("--seed", type=int, default=2027)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sources = default_sources()

    print("Extracting phenotypes from masks...")
    cases, ref, pred = build_phenotype_frames(sources, limit=args.limit)
    print(f"  cases={len(cases)}  ref_rows={len(ref)}  pred_rows={len(pred)}")

    print("Building queries and reference relevance...")
    queries, relevance, thresholds = build_queries_relevance(cases, ref, seed=args.seed)
    print(f"  queries={len(queries)}  relevance_rows={len(relevance)}")

    cases.to_csv(out / "cases.csv", index=False)
    ref.to_csv(out / "phenotypes_ref.csv", index=False)
    pred.to_csv(out / "phenotypes_pred.csv", index=False)
    queries.to_csv(out / "queries.csv", index=False)
    relevance.to_csv(out / "relevance.csv", index=False)
    (out / "benchmark_build.json").write_text(json.dumps({
        "sources": [s["spec"].name for s in sources],
        "n_cases": int(len(cases)),
        "split_counts": cases["split"].value_counts().to_dict(),
        "dataset_counts": cases["dataset"].value_counts().to_dict(),
        "thresholds": thresholds,
        "seed": args.seed,
    }, indent=2))
    print("Wrote 5 CSVs + benchmark_build.json to", out.resolve())


if __name__ == "__main__":
    main()
