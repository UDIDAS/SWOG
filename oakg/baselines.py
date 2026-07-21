"""Strong coverage-aware missing-data baselines (Section 5).

Every baseline takes the query row index, candidate row indices, the masked
feature matrix ``X``, the observation mask ``M``, and the fitted ``Corpus``.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .data import Corpus


def cosine_pair(xq: np.ndarray, xj: np.ndarray) -> float:
    nq = np.linalg.norm(xq)
    nj = np.linalg.norm(xj)
    if nq == 0 or nj == 0:
        return 0.0
    return float(np.dot(xq, xj) / (nq * nj))


def zero_imputed_similarity(q_idx, cand_idx, X, M, corpus: Corpus) -> np.ndarray:
    xq = np.nan_to_num(X[q_idx], nan=0.0)
    xc = np.nan_to_num(X[cand_idx], nan=0.0)
    return cosine_similarity(xq[None, :], xc)[0]


def mean_imputed_similarity(q_idx, cand_idx, X, M, corpus: Corpus) -> np.ndarray:
    fill = np.array([
        corpus.feature_means[j] if corpus.feature_type[f] == "numeric" else corpus.feature_modes[j]
        for j, f in enumerate(corpus.features)
    ])
    xq = np.where(M[q_idx], X[q_idx], fill)
    xc = np.where(M[cand_idx], X[cand_idx], fill)
    return cosine_similarity(xq[None, :], xc)[0]


def missing_indicator_similarity(q_idx, cand_idx, X, M, corpus: Corpus) -> np.ndarray:
    xq = np.nan_to_num(X[q_idx], nan=0.0)
    xc = np.nan_to_num(X[cand_idx], nan=0.0)
    zq = np.concatenate([xq, M[q_idx].astype(float)])
    zc = np.concatenate([xc, M[cand_idx].astype(float)], axis=1)
    return cosine_similarity(zq[None, :], zc)[0]


def masked_cosine_similarity(q_idx, cand_idx, X, M, corpus: Corpus) -> np.ndarray:
    """The most important simple baseline: compare only jointly observed features."""
    scores = np.full(len(cand_idx), np.nan)
    for i, j in enumerate(cand_idx):
        joint = M[q_idx] & M[j]
        if not np.any(joint):
            continue
        scores[i] = cosine_pair(X[q_idx, joint], X[j, joint])
    return scores


def gower_similarity(q_idx, cand_idx, X, M, corpus: Corpus) -> np.ndarray:
    ranges = corpus.feature_ranges
    out = np.full(len(cand_idx), np.nan)
    for i, j in enumerate(cand_idx):
        joint = M[q_idx] & M[j]
        if not np.any(joint):
            continue
        sims = []
        for col in np.flatnonzero(joint):
            if corpus.feature_type[corpus.features[col]] == "numeric":
                sims.append(1.0 - min(1.0, abs(X[q_idx, col] - X[j, col]) / ranges[col]))
            else:
                sims.append(float(X[q_idx, col] == X[j, col]))
        out[i] = float(np.mean(sims))
    return out


BASELINE_FUNCTIONS: dict[str, Callable[..., np.ndarray]] = {
    "Zero imputation": zero_imputed_similarity,
    "Mean imputation": mean_imputed_similarity,
    "Missingness indicators": missing_indicator_similarity,
    "Masked cosine": masked_cosine_similarity,
    "Gower": gower_similarity,
}
