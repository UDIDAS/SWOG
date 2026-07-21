"""Selective retrieval and risk-coverage analysis (Section 7)."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .benchmark import candidate_pool
from .config import Config
from .data import Corpus, ExperimentData
from .masking import MaskingRealization
from .metrics import query_metric_row
from .oakg import oakg_scores


def selective_curve(
    X: np.ndarray,
    M: np.ndarray,
    realization: MaskingRealization,
    data: ExperimentData,
    corpus: Corpus,
    config: Config,
    thresholds: Sequence[float] | None = None,
    track: str = "ref",
) -> pd.DataFrame:
    """Sweep gamma_min; report coverage, served nDCG, selective risk, and AURC."""
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 21)
    rows = []
    for gamma_min in thresholds:
        query_rows = []
        for query in data.queries.itertuples(index=False):
            candidates = candidate_pool(query.query_case_id, corpus)
            scores, eligible = oakg_scores(
                query.query_case_id, candidates, X, M, realization, corpus,
                policy="threshold", gamma_min=float(gamma_min),
            )
            valid = eligible & np.isfinite(scores)
            query_rows.append(
                query_metric_row(
                    query.query_id,
                    f"OAKG-threshold-{gamma_min:.2f}",
                    np.asarray(candidates)[valid],
                    scores[valid],
                    data.relevance,
                    served=bool(np.any(valid)),
                    k=config.top_k,
                )
            )
        qdf = pd.DataFrame(query_rows)
        served = qdf["served"].mean()
        ndcg = qdf.loc[qdf["served"], "nDCG@10"].mean()
        rows.append({
            "track": track,
            "gamma_min": gamma_min,
            "coverage": served,
            "nDCG@10_served": ndcg,
            "risk": 1.0 - ndcg if np.isfinite(ndcg) else np.nan,
        })
    out = pd.DataFrame(rows).sort_values("coverage")
    valid = out[["coverage", "risk"]].dropna()
    out["AURC"] = np.trapz(valid["risk"], valid["coverage"]) if len(valid) > 1 else np.nan
    return out
