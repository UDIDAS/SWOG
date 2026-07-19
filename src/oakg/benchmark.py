"""Generic benchmark runner across masking regimes and input tracks."""
from __future__ import annotations

import json
from typing import Sequence

import pandas as pd
import numpy as np

from .baselines import BASELINE_FUNCTIONS
from .config import Config
from .data import Corpus, ExperimentData
from .graph import wl_embeddings
from .masking import MaskingRealization, apply_mask
from .metrics import query_metric_row
from .neural import apply_pairwise_observability_filter, embedding_scores
from .oakg import oakg_scores

OAKG_POLICIES = ["similarity", "product", "threshold", "lexicographic"]


def candidate_pool(query_case_id: str, corpus: Corpus) -> list[str]:
    """Exclude the query case. Add patient-group exclusions here when required."""
    return [c for c in corpus.case_order if c != query_case_id]


def evaluate_vector_methods(
    X: np.ndarray,
    M: np.ndarray,
    track: str,
    realization: MaskingRealization,
    data: ExperimentData,
    corpus: Corpus,
    config: Config,
    gamma_min: float = 0.25,
) -> pd.DataFrame:
    rows = []
    meta = json.dumps(realization.metadata, sort_keys=True)
    for query in data.queries.itertuples(index=False):
        candidates = candidate_pool(query.query_case_id, corpus)
        cidx = np.array([corpus.case_to_row[c] for c in candidates], dtype=int)
        qidx = corpus.case_to_row[query.query_case_id]

        for method, fn in BASELINE_FUNCTIONS.items():
            scores = fn(qidx, cidx, X, M, corpus)
            valid = np.isfinite(scores)
            row = query_metric_row(
                query.query_id,
                f"{method} [{track}]",
                np.asarray(candidates)[valid],
                scores[valid],
                data.relevance,
                served=bool(np.any(valid)),
                k=config.top_k,
            )
            row.update({"track": track, "masking_regime": realization.name, "mask_metadata": meta})
            rows.append(row)

        for policy in OAKG_POLICIES:
            scores, eligible = oakg_scores(
                query.query_case_id, candidates, X, M, realization, corpus,
                policy=policy, gamma_min=gamma_min,
            )
            valid = eligible & np.isfinite(scores)
            row = query_metric_row(
                query.query_id,
                f"OAKG-{policy} [{track}]",
                np.asarray(candidates)[valid],
                scores[valid],
                data.relevance,
                served=bool(np.any(valid)),
                k=config.top_k,
            )
            row.update({"track": track, "masking_regime": realization.name, "mask_metadata": meta})
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate_graph_methods(
    X: np.ndarray,
    M: np.ndarray,
    track: str,
    realization: MaskingRealization,
    data: ExperimentData,
    corpus: Corpus,
    config: Config,
    gamma_min: float = 0.0,
) -> pd.DataFrame:
    """WL graph-kernel backbone (Section 8.1) with Base and +Obs variants.

    Base = cosine over WL embeddings of the masked case-graph (coverage-blind).
    +Obs = same, with OAKG pairwise eligibility (cross-organ pairs dropped).
    """
    rows = []
    meta = json.dumps(realization.metadata, sort_keys=True)
    wl_emb, _ = wl_embeddings(X, M, corpus)
    wl = {c: wl_emb[i] for i, c in enumerate(corpus.case_order)}
    for query in data.queries.itertuples(index=False):
        candidates = candidate_pool(query.query_case_id, corpus)
        base = embedding_scores(query.query_case_id, candidates, wl)
        obs, elig = apply_pairwise_observability_filter(
            query.query_case_id, candidates, base, realization, gamma_min=gamma_min)
        for method, scores, valid in [
            (f"WL [{track}]", base, np.isfinite(base)),
            (f"WL+Obs [{track}]", obs, elig & np.isfinite(obs)),
        ]:
            row = query_metric_row(
                query.query_id, method,
                np.asarray(candidates)[valid], scores[valid],
                data.relevance, served=bool(np.any(valid)), k=config.top_k)
            row.update({"track": track, "masking_regime": realization.name, "mask_metadata": meta})
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate_realization(
    realization: MaskingRealization,
    data: ExperimentData,
    corpus: Corpus,
    config: Config,
) -> pd.DataFrame:
    ref_x, ref_m = apply_mask(corpus.x_ref_full, corpus.m_ref_full, realization, corpus)
    pred_x, pred_m = apply_mask(corpus.x_pred_full, corpus.m_pred_full, realization, corpus)
    frames = [
        evaluate_vector_methods(ref_x, ref_m, "ref", realization, data, corpus, config),
        evaluate_vector_methods(pred_x, pred_m, "pred", realization, data, corpus, config),
    ]
    if config.run_wl_baseline:
        frames.append(evaluate_graph_methods(ref_x, ref_m, "ref", realization, data, corpus, config))
        frames.append(evaluate_graph_methods(pred_x, pred_m, "pred", realization, data, corpus, config))
    return pd.concat(frames, ignore_index=True)


def run_full_benchmark(
    realizations: Sequence[MaskingRealization],
    data: ExperimentData,
    corpus: Corpus,
    config: Config,
) -> pd.DataFrame:
    frames = [evaluate_realization(r, data, corpus, config) for r in realizations]
    return pd.concat(frames, ignore_index=True)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["track", "masking_regime", "mask_metadata", "method"])
        [["P@10", "R@10", "AP", "nDCG@10", "served"]]
        .mean()
        .reset_index()
        .rename(columns={"AP": "mAP", "served": "ServedRate"})
    )
