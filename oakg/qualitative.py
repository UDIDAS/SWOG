"""Qualitative retrieval examples — complex multi-condition queries + OAKG top-5.

Produces a small, GitHub-friendly artifact under results/qualitative/ so the team
can (a) view the example queries and their retrievals, and (b) run their own:
edit results/qualitative/queries.json (or pass --queries) and re-run.

    python -m oakg.qualitative            # default queries
    python -m oakg.qualitative --queries my_queries.json

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
          "python -m oakg.qualitative        # edit results/qualitative/queries.json first",
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


def imputation_contrast(config: Config, out_dir: Path, targets=("pancreas", "liver", "spleen", "left_kidney", "right_kidney")):
    """Side-by-side where imputation FAILS: hard-distractor queries.

    Query = a broad (multi-organ) case with a large target organ. Pool = narrow
    single-organ cases that truly match (relevant) + broad cases with a SMALL
    target organ (distractors). Imputation ranks the broad distractors high
    (shared non-target organs); OAKG restricts to the shared target evidence and
    returns the true matches. Only targets with both narrow-relevant and
    broad-distractor cases yield a contrast.
    """
    from .baselines import zero_imputed_similarity
    data = load_experiment_data(config.data_dir); corpus = Corpus.build(data, config)
    ds = data.cases.set_index("case_id")["dataset"].to_dict()
    org = {r.case_id: set(str(r.available_organs).split("|")) for r in data.cases.itertuples(index=False)}
    split = {r.case_id: r.split for r in data.cases.itertuples(index=False)}
    real = make_uniform_mask(data.cases, config.organs, config.seed)
    X, M = apply_mask(corpus.x_ref_full, corpus.m_ref_full, real, corpus)
    val = lambda c, f: corpus.x_ref_full[corpus.case_to_row[c], corpus.feature_index[f]]

    rows = []; qn = 0
    for TARGET in targets:
        FEAT = f"{TARGET}_volume_cm3"
        if FEAT not in corpus.feature_index:
            continue
        fit = [c for c in corpus.case_order if split[c] in ("train", "val") and TARGET in org[c]]
        vv = [val(c, FEAT) for c in fit if np.isfinite(val(c, FEAT))]
        if not vv:
            continue
        thr = float(np.median(vv))
        broad = [c for c in corpus.case_order if len(org[c]) >= 4]
        narrow = [c for c in corpus.case_order if org[c] == {TARGET}]
        relevant = [c for c in narrow if np.isfinite(val(c, FEAT)) and val(c, FEAT) >= thr]
        anchors = [c for c in broad if split[c] == "test" and TARGET in org[c]
                   and np.isfinite(val(c, FEAT)) and val(c, FEAT) >= thr]
        for anchor in anchors:
            distract = [c for c in broad if c != anchor and np.isfinite(val(c, FEAT)) and val(c, FEAT) < thr]
            if not relevant or not distract:
                continue
            pool = relevant + distract; relset = set(relevant)
            ci = np.array([corpus.case_to_row[c] for c in pool]); qi = corpus.case_to_row[anchor]
            imp = zero_imputed_similarity(qi, ci, X, M, corpus)
            oak = oakg_scores(anchor, pool, X, M, real, corpus, policy="similarity")[0]
            qn += 1; qid = f"HD{qn}"
            imp_top = rank_ids(pool, imp)[:5]; oak_top = rank_ids(pool, oak)[:5]
            for rank in range(5):
                ic, oc = imp_top[rank], oak_top[rank]
                rows.append({
                    "query_id": qid, "query_case": anchor, "target": TARGET, "threshold_cm3": round(thr, 0),
                    "rank": rank + 1,
                    "imputation_case": ic, "imp_coverage": len(org[ic]), "imp_target_cm3": round(val(ic, FEAT), 0),
                    "imp_relevant": ic in relset,
                    "oakg_case": oc, "oakg_coverage": len(org[oc]), "oakg_target_cm3": round(val(oc, FEAT), 0),
                    "oakg_relevant": oc in relset,
                })
            if qn >= 6:
                break
        if qn >= 6:
            break

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "imputation_vs_oakg.csv", index=False)
    _write_contrast_md(df, out_dir / "imputation_vs_oakg.md")
    if len(df):
        ip = df["imp_relevant"].mean(); op = df["oakg_relevant"].mean()
        print(f"wrote {out_dir}/imputation_vs_oakg.*  ({df['query_id'].nunique()} queries; "
              f"top-5 relevant: imputation={ip:.0%}  OAKG={op:.0%})")
    return df


def _write_contrast_md(df: pd.DataFrame, path: Path):
    md = ["# Where zero-imputation fails — imputation vs OAKG (hard-distractor)",
          "",
          "Query = a broad multi-organ case with a large target organ. Imputation ranks",
          "broad cases high because they share *other* organs — even when their target",
          "organ is small (wrong). OAKG restricts to the shared target evidence and",
          "returns the true matches. ✓ = correct (relevant), ✗ = wrong.", ""]
    if len(df):
        md.append(f"Top-5 relevant rate:  **imputation {df['imp_relevant'].mean():.0%}**  vs  **OAKG {df['oakg_relevant'].mean():.0%}**.\n")
    for qid, grp in df.groupby("query_id"):
        r0 = grp.iloc[0]
        md.append(f"### {qid}: query `{r0.query_case}` — large **{r0.target}** (≥ {int(r0.threshold_cm3)} cm³)")
        md.append("")
        md.append("| rank | imputation → | | OAKG → | |")
        md.append("|---|---|---|---|---|")
        for r in grp.itertuples():
            im = "✓" if r.imp_relevant else "✗"; om = "✓" if r.oakg_relevant else "✗"
            md.append(f"| {r.rank} | {r.imputation_case} ({r.imp_coverage}-organ, {r.target[:4]}={int(r.imp_target_cm3)}) | {im} "
                      f"| {r.oakg_case} ({r.oakg_coverage}-organ, {r.target[:4]}={int(r.oakg_target_cm3)}) | {om} |")
        md.append("")
    path.write_text("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/qualitative")
    ap.add_argument("--queries", default=None, help="path to a queries.json to use instead of defaults")
    ap.add_argument("--contrast", action="store_true", help="also write the imputation-vs-OAKG hard-distractor contrast")
    args = ap.parse_args()
    cfg = Config(use_demo_data=False, data_dir=args.data)
    if args.queries:
        needs = json.loads(Path(args.queries).read_text())
    else:
        default_json = Path(args.out) / "queries.json"
        needs = json.loads(default_json.read_text()) if default_json.exists() else DEFAULT_NEEDS
    run(cfg, needs, Path(args.out))
    if args.contrast:
        imputation_contrast(cfg, Path(args.out))


if __name__ == "__main__":
    main()
