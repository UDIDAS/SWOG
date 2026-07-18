"""Full-to-partial ranking consistency (Section 10.1).

Each method is compared only with its own complete-input ranking from the same
input track (never a predicted-partial ranking against a reference-complete one).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from .benchmark import candidate_pool
from .config import TOP_K
from .data import Corpus, ExperimentData
from .masking import MaskingRealization
from .oakg import oakg_scores


def top_k_overlap(ids_a: Sequence[str], ids_b: Sequence[str], k: int = TOP_K) -> float:
    return len(set(ids_a[:k]) & set(ids_b[:k])) / k


def rank_ids(candidate_ids: Sequence[str], scores: np.ndarray) -> list[str]:
    valid = np.isfinite(scores)
    ids = np.asarray(candidate_ids)[valid]
    vals = scores[valid]
    order = np.argsort(-vals, kind="mergesort")
    return ids[order].tolist()


def ranking_consistency(
    full_X: np.ndarray,
    full_M: np.ndarray,
    partial_X: np.ndarray,
    partial_M: np.ndarray,
    realization_full: MaskingRealization,
    realization_partial: MaskingRealization,
    track: str,
    data: ExperimentData,
    corpus: Corpus,
    policy: str = "product",
    k: int = TOP_K,
) -> pd.DataFrame:
    rows = []
    for query in data.queries.itertuples(index=False):
        candidates = candidate_pool(query.query_case_id, corpus)
        full_scores, _ = oakg_scores(
            query.query_case_id, candidates, full_X, full_M, realization_full, corpus, policy=policy
        )
        partial_scores, _ = oakg_scores(
            query.query_case_id, candidates, partial_X, partial_M, realization_partial, corpus, policy=policy
        )
        full_rank = rank_ids(candidates, full_scores)
        partial_rank = rank_ids(candidates, partial_scores)
        common = [c for c in full_rank if c in set(partial_rank)]
        if len(common) >= 2:
            full_pos = {c: i for i, c in enumerate(full_rank)}
            partial_pos = {c: i for i, c in enumerate(partial_rank)}
            x = [full_pos[c] for c in common]
            y = [partial_pos[c] for c in common]
            tau = kendalltau(x, y).statistic
            rho = spearmanr(x, y).statistic
        else:
            tau = np.nan
            rho = np.nan
        rows.append({
            "query_id": query.query_id,
            "track": track,
            "kendall_tau": tau,
            "spearman_rho": rho,
            "top10_overlap": top_k_overlap(full_rank, partial_rank, k),
        })
    return pd.DataFrame(rows)
