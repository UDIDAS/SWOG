"""Build the five OAKG benchmark CSVs from paired GT/predicted segmentation masks.

Three heterogeneous sources populate one MMKG-shaped benchmark:
  - Pancreas, LiTS: single-organ, organ + tumor observations.
  - FLARE:          multi-organ (5 organs), organ morphometry only (no tumor).
FLARE is the multi-organ hub: its cases share organs with both single-organ
datasets, so the shared-evidence coefficient gamma becomes non-degenerate and
cross-organ retrieval is possible. Reference (GT) masks populate the ref track
and define relevance; predicted masks populate the end-to-end pred track.

Run:  PYTHONPATH=src python -m oakg.build_benchmark --out data [--limit N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .phenotypes import DatasetSpec, organ_phenotypes

ACM = Path("/scratch/ud3d4/acm_data/ssl_handoff_ours")
FLARE_ROOT = Path("/scratch/ud3d4/acm_data/FLARE_Task2")
FLARE_SAM = FLARE_ROOT / "sam3_delivery"
FLARE_ORGANS = ("liver", "pancreas", "spleen", "left_kidney", "right_kidney")
# Validated against per-organ GT (Dice=1.000): FLARE22 multi-label integers.
FLARE_LABELS = {"liver": 1, "right_kidney": 2, "spleen": 3, "pancreas": 4, "left_kidney": 13}


def _combined(dirs, organ_labels, tumor_labels=None):
    return {"kind": "combined", "dirs": [Path(d) for d in dirs],
            "organ_labels": organ_labels, "tumor_labels": tumor_labels or {}}


def _per_organ(base_dir, suffix):
    return {"kind": "per_organ", "base_dir": Path(base_dir), "suffix": suffix}


def default_sources() -> list[dict]:
    return [
        {
            "spec": DatasetSpec("Pancreas", ("pancreas",),
                                organ_labels={"pancreas": 1}, tumor_labels={"pancreas": 2}),
            "ref": _combined([ACM / "ground_truth" / "pancreas"], {"pancreas": 1}, {"pancreas": 2}),
            "pred": _combined([ACM / "ssl_predictions" / "pancreas"], {"pancreas": 1}, {"pancreas": 2}),
        },
        {
            "spec": DatasetSpec("LiTS", ("liver",),
                                organ_labels={"liver": 1}, tumor_labels={"liver": 2}),
            "ref": _combined([ACM / "ground_truth" / "lits"], {"liver": 1}, {"liver": 2}),
            "pred": _combined([ACM / "ssl_predictions" / "lits"], {"liver": 1}, {"liver": 2}),
        },
        {
            # FLARE: 100 multi-organ GT cases (labelsTr + validation, 13-label),
            # predictions for the 20 in sam3_delivery (per-organ binary). No tumor.
            "spec": DatasetSpec("FLARE", FLARE_ORGANS),
            "ref": [
                _combined([FLARE_ROOT / "train_gt_label" / "labelsTr",
                           FLARE_ROOT / "validation" / "Validation-Public-Labels"], FLARE_LABELS),
                _per_organ(FLARE_SAM, "gt"),   # fallback for any sam3-only case
            ],
            "pred": _per_organ(FLARE_SAM, "pred"),
        },
    ]


def _read_mask(path: Path):
    import nibabel as nib
    img = nib.load(str(path))
    arr = np.asanyarray(img.dataobj)
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    return arr, spacing


def _resolve(access, case_id: str, organs) -> dict[str, dict] | None:
    """Load {organ: phenotypes} for one case via an access descriptor (or list of them)."""
    for acc in (access if isinstance(access, list) else [access]):
        if acc["kind"] == "combined":
            path = next((d / f"{case_id}.nii.gz" for d in acc["dirs"]
                         if (d / f"{case_id}.nii.gz").exists()), None)
            if path is None:
                continue
            mask, sp = _read_mask(path)
            return {o: organ_phenotypes(mask, sp, acc["organ_labels"][o], acc["tumor_labels"].get(o))
                    for o in organs}
        else:  # per_organ
            base, suf = acc["base_dir"], acc["suffix"]
            paths = {o: base / o / f"{case_id}_{suf}.nii.gz" for o in organs}
            if not all(p.exists() for p in paths.values()):
                continue
            out = {}
            for o, p in paths.items():
                mask, sp = _read_mask(p)
                out[o] = organ_phenotypes(mask, sp, organ_label=1, tumor_label=None)
            return out
    return None


def _case_phenotypes(src: dict, case_id: str, track: str) -> dict[str, dict] | None:
    return _resolve(src[track], case_id, src["spec"].organs)


def _feature_rows(spec: DatasetSpec, organ_ph: dict[str, dict]) -> list[tuple[str, str, str, float]]:
    rows: list[tuple[str, str, str, float]] = []
    for organ in spec.organs:
        ph = organ_ph[organ]
        rows.append((f"{organ}_present", "binary", organ, ph["present"]))
        if ph["volume_cm3"] is not None:
            rows.append((f"{organ}_volume_cm3", "numeric", organ, ph["volume_cm3"]))
        if organ in spec.tumor_labels:
            rows.append((f"{organ}_tumor_present", "binary", organ, ph["tumor_present"]))
            if ph["tumor_containment"] is not None:
                rows.append((f"{organ}_tumor_containment", "binary", organ, ph["tumor_containment"]))
    if spec.has_tumor:
        tumor_organs = [o for o in spec.organs if o in spec.tumor_labels]
        has = any(organ_ph[o]["tumor_present"] for o in tumor_organs)
        burden = sum(organ_ph[o]["tumor_burden_cm3"] or 0.0 for o in tumor_organs)
        mult = sum(organ_ph[o]["lesion_multiplicity"] or 0.0 for o in tumor_organs)
        rows += [
            ("has_tumor", "binary", "", float(has)),
            ("tumor_burden_cm3", "numeric", "", float(burden)),
            ("lesion_multiplicity", "numeric", "", float(mult)),
        ]
    return [(f, t, s, float(v)) for (f, t, s, v) in rows if v is not None]


def _case_ids(src: dict) -> list[str]:
    """Union of case ids reachable through the source's ref access descriptor(s)."""
    ids: set[str] = set()
    organs = src["spec"].organs
    for acc in (src["ref"] if isinstance(src["ref"], list) else [src["ref"]]):
        if acc["kind"] == "combined":
            for d in acc["dirs"]:
                ids |= {p.name.replace(".nii.gz", "") for p in Path(d).glob("*.nii.gz")}
        else:
            anchor = Path(acc["base_dir"]) / organs[0]
            ids |= {p.name.replace(f"_{acc['suffix']}.nii.gz", "") for p in anchor.glob(f"*_{acc['suffix']}.nii.gz")}
    return sorted(ids)


def _assign_split(case_id: str, ratios=(0.55, 0.20, 0.25)) -> str:
    """Deterministic patient-level split from a stable hash of the case id."""
    h = int(hashlib.md5(case_id.encode()).hexdigest(), 16) % 10_000 / 10_000
    if h < ratios[0]:
        return "train"
    if h < ratios[0] + ratios[1]:
        return "val"
    return "test"


def build_phenotype_frames(sources: list[dict], limit: int | None = None):
    case_rows, ref_rows, pred_rows = [], [], []
    for src in sources:
        spec: DatasetSpec = src["spec"]
        ids = _case_ids(src)
        if limit:
            ids = ids[:limit]
        for case_id in ids:
            ref_ph = _case_phenotypes(src, case_id, "ref")
            if ref_ph is None:
                continue
            for f, t, s, v in _feature_rows(spec, ref_ph):
                ref_rows.append({"case_id": case_id, "feature": f, "value": v,
                                 "feature_type": t, "support_organs": s})
            pred_ph = _case_phenotypes(src, case_id, "pred")
            if pred_ph is not None:
                for f, t, s, v in _feature_rows(spec, pred_ph):
                    pred_rows.append({"case_id": case_id, "feature": f, "value": v,
                                      "feature_type": t, "support_organs": s})
            case_rows.append({
                "case_id": case_id,
                "dataset": spec.name,
                "split": _assign_split(case_id),
                "available_organs": "|".join(spec.organs),
            })
    return (pd.DataFrame(case_rows), pd.DataFrame(ref_rows), pd.DataFrame(pred_rows))


# --- Queries and relevance (reference-defined) -----------------------------
def _fit_numeric_thresholds(cases, wide, numeric_feats):
    fit_ids = cases.loc[cases["split"].isin(["train", "val"]), "case_id"]
    thr = {}
    for f in numeric_feats:
        vals = wide.reindex(fit_ids)[f].dropna() if f in wide.columns else pd.Series(dtype=float)
        thr[f] = float(np.quantile(vals, 0.55)) if len(vals) else 0.0
    thr["lesion_multiplicity"] = 2.0  # fixed clinically meaningful cut
    return thr


def build_queries_relevance(cases: pd.DataFrame, ref: pd.DataFrame, seed: int = 2027):
    rng = np.random.default_rng(seed)
    wide = ref.pivot_table(index="case_id", columns="feature", values="value")
    ftype = dict(zip(ref["feature"], ref["feature_type"]))
    support_of = dict(zip(ref["feature"], ref["support_organs"].fillna("")))
    numeric_feats = [f for f, t in ftype.items() if t == "numeric"]
    thresholds = _fit_numeric_thresholds(cases, wide, numeric_feats)

    all_ids = cases["case_id"].tolist()
    test_ids = cases.loc[cases["split"].eq("test"), "case_id"].tolist()

    def support_list(feat: str) -> list[str]:
        s = support_of.get(feat, "")
        return [s] if s else []

    def make_predicate(feat: str, case_id: str) -> dict:
        if ftype.get(feat) == "numeric":
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
        return v >= pred["value"] if pred["op"] == ">=" else v == pred["value"]

    query_rows, rel_rows = [], []
    qid = 0
    for qcase in test_ids:
        observed = [f for f in wide.columns if not pd.isna(wide.at[qcase, f])]
        # Discriminative query features: numeric metrics + tumor-presence flags.
        pool = [f for f in observed
                if f in numeric_feats or f.endswith("_tumor_present") or f == "has_tumor"]
        if not pool:
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
    ap.add_argument("--out", default="data")
    ap.add_argument("--limit", type=int, default=None, help="cap cases per dataset (debug)")
    ap.add_argument("--seed", type=int, default=2027)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sources = default_sources()

    print("Extracting phenotypes from masks...")
    cases, ref, pred = build_phenotype_frames(sources, limit=args.limit)
    print(f"  cases={len(cases)}  ref_rows={len(ref)}  pred_rows={len(pred)}")
    print("  by dataset:", cases["dataset"].value_counts().to_dict())

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
        "features": sorted(ref["feature"].unique().tolist()),
        "thresholds": thresholds,
        "seed": args.seed,
    }, indent=2))
    print("Wrote 5 CSVs + benchmark_build.json to", out.resolve())


if __name__ == "__main__":
    main()
