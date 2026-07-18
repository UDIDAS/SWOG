"""Neural backbones: CompGCN/CT embedding I/O, observed-region CT, hybrid, cross-backbone.

Covers Sections 8.2-8.4, 9. Embeddings are consumed as precomputed ``.npz``
files (keys ``case_ids`` and ``embeddings``) exported by the training code, so
retrieval evaluation is reproducible without re-running the encoders.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .masking import MaskingRealization
from .metrics import ndcg_at_k
from .oakg import observation_overlap


# --- Embedding I/O ---------------------------------------------------------
def load_npz_embeddings(path: Path) -> dict[str, np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    ids = payload["case_ids"].astype(str)
    emb = payload["embeddings"].astype(float)
    if emb.ndim != 2 or len(ids) != len(emb):
        raise ValueError(f"Invalid embedding file: {path}")
    return {cid: vector for cid, vector in zip(ids, emb)}


def embedding_scores(
    query_case: str,
    candidate_ids: Sequence[str],
    embeddings: Mapping[str, np.ndarray],
) -> np.ndarray:
    q = embeddings[query_case]
    matrix = np.stack([embeddings[c] for c in candidate_ids])
    return cosine_similarity(q[None, :], matrix)[0]


# --- Observed-region CT (Section 8.3) --------------------------------------
def restrict_ct_volume(
    volume: np.ndarray,
    observed_region_mask: np.ndarray,
    background_value: float = 0.0,
) -> np.ndarray:
    if volume.shape != observed_region_mask.shape:
        raise ValueError("Volume and observation mask must have the same shape.")
    return np.where(observed_region_mask.astype(bool), volume, background_value)


def masked_global_pool(feature_map: np.ndarray, pooled_mask: np.ndarray) -> np.ndarray:
    """Pool a channel-first feature map (C, D, H, W) over an observed-region mask."""
    if feature_map.ndim != 4:
        raise ValueError("Expected feature map shape (C, D, H, W).")
    if feature_map.shape[1:] != pooled_mask.shape:
        raise ValueError("Feature map spatial shape and mask shape do not match.")
    locations = pooled_mask.astype(bool)
    if not np.any(locations):
        return np.zeros(feature_map.shape[0], dtype=float)
    return feature_map[:, locations].mean(axis=1)


# --- Hybrid image-graph retrieval (Section 8.4) ----------------------------
def zscore_fit(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    return (
        float(np.mean(finite)) if len(finite) else 0.0,
        float(np.std(finite)) if len(finite) and np.std(finite) > 0 else 1.0,
    )


def zscore_apply(values: np.ndarray, stats: tuple[float, float]) -> np.ndarray:
    mean, std = stats
    return (values - mean) / std


def hybrid_scores(
    graph_scores: np.ndarray,
    image_scores: np.ndarray,
    alpha: float,
    graph_stats: tuple[float, float],
    image_stats: tuple[float, float],
) -> np.ndarray:
    return (
        alpha * zscore_apply(graph_scores, graph_stats)
        + (1.0 - alpha) * zscore_apply(image_scores, image_stats)
    )


def select_hybrid_alpha(
    validation_rows: pd.DataFrame,
    alpha_grid: Sequence[float] | None = None,
) -> float:
    """Select alpha on validation queries by nDCG@10 (validation statistics only)."""
    if alpha_grid is None:
        alpha_grid = np.linspace(0.0, 1.0, 11)
    required = {"query_id", "candidate_id", "graph_score", "image_score", "graded_relevance"}
    if not required.issubset(validation_rows.columns):
        raise ValueError(f"Validation frame must contain {sorted(required)}")

    gstats = zscore_fit(validation_rows["graph_score"].to_numpy())
    istats = zscore_fit(validation_rows["image_score"].to_numpy())
    rows = []
    for alpha in alpha_grid:
        scores = hybrid_scores(
            validation_rows["graph_score"].to_numpy(),
            validation_rows["image_score"].to_numpy(),
            float(alpha), gstats, istats,
        )
        frame = validation_rows.assign(hybrid_score=scores)
        query_scores = [
            ndcg_at_k(q["graded_relevance"].to_numpy(), q["hybrid_score"].to_numpy())
            for _, q in frame.groupby("query_id")
        ]
        rows.append((float(alpha), float(np.nanmean(query_scores))))
    return max(rows, key=lambda x: x[1])[0]


# --- Cross-backbone observability (Section 9) ------------------------------
def apply_pairwise_observability_filter(
    query_case: str,
    candidate_ids: Sequence[str],
    scores: np.ndarray,
    realization: MaskingRealization,
    gamma_min: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """+Obs filter: keep only pairs with shared observed evidence >= gamma_min."""
    eligible = np.zeros(len(candidate_ids), dtype=bool)
    adjusted = scores.copy().astype(float)
    for i, cand in enumerate(candidate_ids):
        inter, gamma = observation_overlap(query_case, cand, realization)
        eligible[i] = bool(inter) and gamma >= gamma_min
        if not eligible[i]:
            adjusted[i] = np.nan
    return adjusted, eligible


def cross_backbone_gain(
    query_results: pd.DataFrame,
    base_method: str,
    obs_method: str,
    metric: str = "nDCG@10",
) -> dict[str, float]:
    """Delta_obs = Metric(base+Obs) - Metric(base), holding evidence constant."""
    pivot = query_results.pivot_table(index="query_id", columns="method", values=metric)
    pair = pivot[[base_method, obs_method]].dropna()
    diff = pair[obs_method] - pair[base_method]
    return {
        "backbone": base_method,
        "n_queries": len(pair),
        f"Delta_obs_{metric}": float(diff.mean()) if len(diff) else np.nan,
    }
