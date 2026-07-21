"""Qualitative retrieval examples — complex multi-condition queries + OAKG top-5.

Produces a small, GitHub-friendly artifact under results/qualitative/ so the team
can (a) view the example queries and their retrievals, and (b) run their own:
edit results/qualitative/queries.json (or pass --queries) and re-run.

    PYTHONPATH=src python -m oakg.qualitative            # default queries
    PYTHONPATH=src python -m oakg.qualitative --queries my_queries.json

Retrieval is content-based (rank all cases by OAKG-product similarity to a
representative anchor case that satisfies the query); relevance is graded by how
many of the query's conditions each retrieved case also satisfies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .data import load_experiment_data, Corpus
from .masking import make_uniform_mask, apply_mask
from .benchmark import candidate_pool
from .oakg import oakg_scores, observation_overlap
from .consistency import rank_ids

TOP_K = 5

# Default example queries. Each: {id, need, conditions:[{feature, op, value}]}.
# op in {"==", ">=", "<="}. Edit / add your own and re-run (or use queries.json).
DEFAULT_NEEDS = [
    {"id": "Q1", "need": "Pancreatic tumour, contained, in a large pancreas",
     "conditions": [{"feature": "pancreas_tumor_present", "op": "==", "value": 1},
                    {"feature": "pancreas_tumor_containment", "op": "==", "value": 1},
                    {"feature": "pancreas_volume_cm3", "op": ">=", "value": 85}]},
    {"id": "Q2", "need": "Liver tumour, multifocal, high tumour burden",
     "conditions": [{"feature": "liver_tumor_present", "op": "==", "value": 1},
                    {"feature": "lesion_multiplicity", "op": ">=", "value": 2},
                    {"feature": "tumor_burden_cm3", "op": ">=", "value": 6}]},
    {"id": "Q3", "need": "Multi-organ: large spleen & liver, sizeable kidneys",
     "conditions": [{"feature": "spleen_volume_cm3", "op": ">=", "value": 245},
                    {"feature": "liver_volume_cm3", "op": ">=", "value": 1152},
                    {"feature": "left_kidney_volume_cm3", "op": ">=", "value": 198},
                    {"feature": "right_kidney_volume_cm3", "op": ">=", "value": 198}]},
    {"id": "Q4", "need": "Multi-organ: sizeable pancreas with balanced kidneys & spleen",
     "conditions": [{"feature": "pancreas_volume_cm3", "op": ">=", "value": 85},
                    {"feature": "left_kidney_volume_cm3", "op": ">=", "value": 198},
                    {"feature": "right_kidney_volume_cm3", "op": ">=", "value": 198},
                    {"feature": "spleen_volume_cm3", "op": ">=", "value": 197}]},
    {"id": "Q5", "need": "Any tumour, multifocal, above-median burden",
     "conditions": [{"feature": "has_tumor", "op": "==", "value": 1},
                    {"feature": "lesion_multiplicity", "op": ">=", "value": 2},
                    {"feature": "tumor_burden_cm3", "op": ">=", "value": 5}]},
    {"id": "Q6", "need": "Pancreatic tumour, contained, low burden (early-stage)",
     "conditions": [{"feature": "pancreas_tumor_present", "op": "==", "value": 1},
                    {"feature": "pancreas_tumor_containment", "op": "==", "value": 1},
                    {"feature": "tumor_burden_cm3", "op": "<=", "value": 5}]},
    {"id": "Q7", "need": "Liver tumour, solitary lesion, above-median liver volume",
     "conditions": [{"feature": "liver_tumor_present", "op": "==", "value": 1},
                    {"feature": "lesion_multiplicity", "op": "<=", "value": 1},
                    {"feature": "liver_volume_cm3", "op": ">=", "value": 1002}]},
    {"id": "Q8", "need": "Multi-organ with a pancreatic tumour present (cross-source)",
     "conditions": [{"feature": "pancreas_present", "op": "==", "value": 1},
                    {"feature": "pancreas_tumor_present", "op": "==", "value": 1},
                    {"feature": "pancreas_volume_cm3", "op": ">=", "value": 70}]},
]


def _sat(corpus, c, cond) -> bool:
    f = cond["feature"]
    if f not in corpus.feature_index:
        return False
    v = corpus.x_ref_full[corpus.case_to_row[c], corpus.feature_index[f]]
    if not np.isfinite(v):
        return False
    op, t = cond["op"], cond["value"]
    return (v == t) if op == "==" else ((v >= t) if op == ">=" else (v <= t))


def _n_sat(corpus, c, conds) -> int:
    return sum(_sat(corpus, c, k) for k in conds)


def run(config: Config, needs: list[dict], out_dir: Path):
    data = load_experiment_data(config.data_dir)
    corpus = Corpus.build(data, config)
    ds = data.cases.set_index("case_id")["dataset"].to_dict()
    real = make_uniform_mask(data.cases, config.organs, config.seed)  # natural coverage
    X, M = apply_mask(corpus.x_ref_full, corpus.m_ref_full, real, corpus)

    rows = []
    for q in needs:
        conds = q["conditions"]; n = len(conds)
        # anchor = a case satisfying the most conditions (ties -> first)
        best = max(corpus.case_order, key=lambda c: _n_sat(corpus, c, conds))
        if _n_sat(corpus, best, conds) == 0:
            continue
        anchor = best
        cands = candidate_pool(anchor, corpus)
        s = oakg_scores(anchor, cands, X, M, real, corpus, policy="product")[0]
        for rank, c in enumerate(rank_ids(cands, s)[:TOP_K], 1):
            vals = {k["feature"]: round(float(corpus.x_ref_full[corpus.case_to_row[c], corpus.feature_index[k["feature"]]]), 1)
                    for k in conds if k["feature"] in corpus.feature_index
                    and np.isfinite(corpus.x_ref_full[corpus.case_to_row[c], corpus.feature_index[k["feature"]]])}
            rows.append({
                "query_id": q["id"], "need": q["need"], "n_conditions": n,
                "anchor": anchor, "rank": rank, "candidate": c, "dataset": ds[c],
                "gamma": round(observation_overlap(anchor, c, real)[1], 2),
                "conditions_matched": _n_sat(corpus, c, conds),
                "query_feature_values": "; ".join(f"{k}={v}" for k, v in vals.items()),
            })
    df = pd.DataFrame(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "qualitative_retrieval.csv", index=False)
    (out_dir / "queries.json").write_text(json.dumps(needs, indent=2))
    _write_markdown(df, out_dir / "qualitative_retrieval.md")
    mean = float((df["conditions_matched"] / df["n_conditions"]).mean())
    print(f"wrote {out_dir}/  ({df['query_id'].nunique()} queries, mean conditions-matched@5 = {mean:.2f})")
    return df


def _write_markdown(df: pd.DataFrame, path: Path):
    md = ["# OAKG — qualitative retrieval examples",
          "",
          "Content-based retrieval on the 512-case benchmark (natural coverage): OAKG's",
          "top-5 for each multi-condition query. ✓ = matches **all** conditions, else the",
          "fraction matched. Regenerate / add your own queries:",
          "",
          "```bash",
          "PYTHONPATH=src python -m oakg.qualitative        # edit results/qualitative/queries.json first",
          "```",
          "",
          f"Mean conditions-matched@5 = **{(df['conditions_matched']/df['n_conditions']).mean():.2f}**.",
          "",
          "| # | Information need | Cond. | OAKG top-5 (dataset · match) |",
          "|---|---|---|---|"]
    for q, grp in df.groupby("query_id"):
        need = grp["need"].iloc[0]; n = int(grp["n_conditions"].iloc[0])
        cells = []
        for r in grp.sort_values("rank").itertuples():
            mark = "✓" if r.conditions_matched == n else f"{r.conditions_matched}/{n}"
            cells.append(f"{r.candidate} ({r.dataset[:4]}·{mark})")
        md.append(f"| {q} | {need} | {n} | " + "; ".join(cells) + " |")
    path.write_text("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/qualitative")
    ap.add_argument("--queries", default=None, help="path to a queries.json to use instead of defaults")
    args = ap.parse_args()
    cfg = Config(use_demo_data=False, data_dir=args.data)
    if args.queries:
        needs = json.loads(Path(args.queries).read_text())
    else:
        default_json = Path(args.out) / "queries.json"
        needs = json.loads(default_json.read_text()) if default_json.exists() else DEFAULT_NEEDS
    run(cfg, needs, Path(args.out))


if __name__ == "__main__":
    main()
