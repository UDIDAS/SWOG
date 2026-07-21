"""Export native (unmasked) candidate-level scores for the cross-dataset analysis.

Produces the four input CSVs the provided oakg_benchmark.native_cross_dataset package
expects (cases, queries, relevance, scores), using the ORIGINAL annotation scopes
(no additional masking) for 5 methods: OAKG, OAKG-Union, ZeroImputation,
MissingnessIndicators, WL.

  OAKG          = support-restricted similarity over the INTERSECTION of native
                  observed organs; comparable=False (score blank) when no organ is
                  shared (incomparable) -> ranked at the bottom downstream.
  OAKG-Union    = similarity over the UNION with one-sided anatomy completed as
                  absent; always comparable.
  Others        = their ordinary similarity; always comparable.

Same-source pairs have matched scope so OAKG == OAKG-Union (built-in neg. control).

Run:  PYTHONPATH=src python -m oakg.native_export --out results/native_inputs
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .data import load_experiment_data, Corpus
from .masking import make_uniform_mask, apply_mask
from .benchmark import candidate_pool
from .baselines import zero_imputed_similarity, missing_indicator_similarity
from .oakg import observation_overlap
from .graph import wl_embeddings
from .neural import embedding_scores
from .union_ablation import _pair_similarity

SOURCE = {"Pancreas": "msd_pancreas", "LiTS": "lits", "FLARE": "flare22"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/native_inputs")
    args = ap.parse_args()
    cfg = Config(use_demo_data=False, data_dir=args.data)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    data = load_experiment_data(cfg.data_dir); corpus = Corpus.build(data, cfg)
    split = dict(zip(data.cases.case_id, data.cases.split))
    real = make_uniform_mask(data.cases, cfg.organs, cfg.seed)          # native scopes
    xf, M = apply_mask(corpus.x_ref_full, corpus.m_ref_full, real, corpus)
    nm = np.array([corpus.feature_type[f] == "numeric" for f in corpus.features])
    ranges = corpus.feature_ranges
    wl_emb, _ = wl_embeddings(xf, M, corpus)
    wl = {c: wl_emb[i] for i, c in enumerate(corpus.case_order)}

    # cases / queries / relevance
    cases = data.cases.assign(source_id=data.cases.dataset.map(SOURCE))[["case_id", "source_id", "split"]]
    cases.to_csv(out / "native_cases.csv", index=False)
    queries = data.queries[["query_id", "query_case_id"]].copy()
    queries["split"] = queries.query_case_id.map(split)
    queries.to_csv(out / "native_queries.csv", index=False)
    rel = data.relevance[["query_id", "candidate_id", "graded_relevance", "binary_relevance"]]
    rel.to_csv(out / "native_relevance.csv", index=False)

    # candidate-level scores over the SAME pool for every method
    rows = []
    qcase = dict(zip(data.queries.query_id, data.queries.query_case_id))
    for qid, qc in qcase.items():
        cands = candidate_pool(qc, corpus)
        cidx = np.array([corpus.case_to_row[c] for c in cands]); qi = corpus.case_to_row[qc]
        zi = zero_imputed_similarity(qi, cidx, xf, M, corpus)
        mi = missing_indicator_similarity(qi, cidx, xf, M, corpus)
        wls = embedding_scores(qc, cands, wl)
        for k, c in enumerate(cands):
            ci = corpus.case_to_row[c]
            inter, _ = observation_overlap(qc, c, real)
            if inter:
                oakg_s = _pair_similarity(qi, ci, xf, M, nm, ranges, "intersection"); oakg_cmp = True
            else:
                oakg_s = None; oakg_cmp = False
            union_s = _pair_similarity(qi, ci, xf, M, nm, ranges, "union")
            for method, score, comp in [
                ("OAKG", oakg_s, oakg_cmp),
                ("OAKG-Union", union_s, True),
                ("ZeroImputation", float(zi[k]), True),
                ("MissingnessIndicators", float(mi[k]), True),
                ("WL", float(wls[k]), True),
            ]:
                rows.append({"query_id": qid, "candidate_id": c, "method": method,
                             "score": "" if score is None else round(float(score), 6),
                             "comparable": comp})
    pd.DataFrame(rows).to_csv(out / "native_scores.csv", index=False)
    inc = pd.DataFrame(rows)
    n_oakg = int((inc.method == "OAKG").sum())
    n_inc = int(((inc.method == "OAKG") & (~inc.comparable)).sum())
    print(f"wrote {out}/  cases={len(cases)} queries={len(queries)} "
          f"score_rows={len(rows)} | OAKG incomparable pairs = {n_inc}/{n_oakg} ({100*n_inc/n_oakg:.1f}%)")


if __name__ == "__main__":
    main()
