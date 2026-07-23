"""Retrieval metrics and per-query scoring rows."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .config import TOP_K


def precision_at_k(y: np.ndarray, scores: np.ndarray, k: int = TOP_K) -> float:
    if len(y) == 0:
        return np.nan
    order = np.argsort(-scores, kind="mergesort")[:k]
    return float(np.mean(y[order] > 0))


def recall_at_k(y: np.ndarray, scores: np.ndarray, k: int = TOP_K) -> float:
    positives = int(np.sum(y > 0))
    if positives == 0:
        return np.nan
    order = np.argsort(-scores, kind="mergesort")[:k]
    return float(np.sum(y[order] > 0) / positives)


def average_precision(y: np.ndarray, scores: np.ndarray) -> float:
    if np.sum(y > 0) == 0:
        return np.nan
    return float(average_precision_score((y > 0).astype(int), scores))


def ndcg_at_k(grades: np.ndarray, scores: np.ndarray, k: int = TOP_K) -> float:
    if len(grades) == 0:
        return np.nan
    order = np.argsort(-scores, kind="mergesort")[:k]
    gains = np.power(2.0, grades[order]) - 1.0
    discounts = np.log2(np.arange(2, len(order) + 2))
    dcg = float(np.sum(gains / discounts))

    ideal = np.argsort(-grades, kind="mergesort")[:k]
    ideal_gains = np.power(2.0, grades[ideal]) - 1.0
    idcg = float(np.sum(ideal_gains / np.log2(np.arange(2, len(ideal) + 2))))
    return dcg / idcg if idcg > 0 else np.nan


def query_metric_row(
    query_id: str,
    method: str,
    candidate_ids: Sequence[str],
    scores: np.ndarray,
    relevance: pd.DataFrame,
    served: bool = True,
    k: int = TOP_K,
) -> dict[str, Any]:
    rel = relevance.loc[relevance["query_id"] == query_id].set_index("candidate_id")
    binary = rel.reindex(candidate_ids)["binary_relevance"].fillna(0).to_numpy(dtype=float)
    graded = rel.reindex(candidate_ids)["graded_relevance"].fillna(0).to_numpy(dtype=float)

    if not served or len(scores) == 0:
        # ALL-QUERY convention: a query the method cannot serve (no comparable
        # candidate) still counts — it scores 0 on the metrics whose relevance
        # class is present (it retrieved nothing useful), and NaN only where the
        # metric is undefined (no relevant candidate at all). This keeps every
        # method averaged over the SAME query set. NaN only if truly no candidates.
        return {
            "query_id": query_id,
            "method": method,
            "P@10": 0.0 if binary.sum() > 0 else np.nan,
            "R@10": 0.0 if binary.sum() > 0 else np.nan,
            "AP": 0.0 if binary.sum() > 0 else np.nan,
            "nDCG@10": 0.0 if graded.sum() > 0 else np.nan,
            "served": False,
        }

    return {
        "query_id": query_id,
        "method": method,
        "P@10": precision_at_k(binary, scores, k),
        "R@10": recall_at_k(binary, scores, k),
        "AP": average_precision(binary, scores),
        "nDCG@10": ndcg_at_k(graded, scores, k),
        "served": True,
    }


def aggregate_metrics(query_results: pd.DataFrame) -> pd.DataFrame:
    metrics = ["P@10", "R@10", "AP", "nDCG@10"]
    summary = (
        query_results.groupby("method")[metrics]
        .mean()
        .rename(columns={"AP": "mAP"})
    )
    served = query_results.groupby("method")["served"].mean().rename("ServedRate")
    return summary.join(served).reset_index()
