"""OAKG support-restricted similarity and ranking policies (Section 6).

OAKG restricts pairwise comparison to jointly observed, anatomically supported
features and modulates ranking by a shared-evidence coefficient ``gamma``.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .data import Corpus
from .masking import MaskingRealization


def observation_overlap(
    q_case: str,
    c_case: str,
    realization: MaskingRealization,
) -> tuple[frozenset[str], float]:
    """Return (shared observed organs, Jaccard shared-evidence coefficient gamma)."""
    oq = realization.observation_sets[q_case]
    oc = realization.observation_sets[c_case]
    inter = oq & oc
    union = oq | oc
    gamma = len(inter) / len(union) if union else 0.0
    return inter, gamma


def oakg_similarity_components(
    q_idx: int,
    c_idx: int,
    X: np.ndarray,
    M: np.ndarray,
    corpus: Corpus,
) -> tuple[float | None, dict[str, float]]:
    joint = M[q_idx] & M[c_idx]
    if not np.any(joint):
        return None, {}

    numeric = np.array([corpus.feature_type[f] == "numeric" for f in corpus.features])
    set_like = ~numeric

    components: dict[str, float] = {}
    numeric_joint = joint & numeric
    if np.any(numeric_joint):
        d = np.abs(X[q_idx, numeric_joint] - X[c_idx, numeric_joint])
        components["numeric"] = float(
            np.mean(np.maximum(0.0, 1.0 - d / corpus.feature_ranges[numeric_joint]))
        )

    set_joint = joint & set_like
    if np.any(set_joint):
        qv = X[q_idx, set_joint]
        cv = X[c_idx, set_joint]
        components["categorical_relational"] = float(np.mean(qv == cv))

    if not components:
        return None, {}
    return float(np.mean(list(components.values()))), components


def rank_policy(
    similarity: float | None,
    gamma: float,
    policy: str,
    gamma_min: float = 0.25,
    evidence_bins: Sequence[float] = (0.25, 0.50, 0.75),
) -> float | tuple[int, float] | None:
    if similarity is None:
        return None
    if policy == "similarity":
        return similarity
    if policy == "product":
        return gamma * similarity
    if policy == "threshold":
        return similarity if gamma >= gamma_min else None
    if policy == "lexicographic":
        category = int(np.digitize(gamma, evidence_bins, right=False))
        return category, similarity
    raise ValueError(policy)


def oakg_scores(
    query_case: str,
    candidate_ids: Sequence[str],
    X: np.ndarray,
    M: np.ndarray,
    realization: MaskingRealization,
    corpus: Corpus,
    policy: str = "product",
    gamma_min: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    q_idx = corpus.case_to_row[query_case]
    raw_scores: list[float] = []
    eligible: list[bool] = []
    for cand in candidate_ids:
        c_idx = corpus.case_to_row[cand]
        inter, gamma = observation_overlap(query_case, cand, realization)
        if not inter:
            raw_scores.append(np.nan)
            eligible.append(False)
            continue
        s, _ = oakg_similarity_components(q_idx, c_idx, X, M, corpus)
        score = rank_policy(s, gamma, policy, gamma_min)
        if score is None:
            raw_scores.append(np.nan)
            eligible.append(False)
        elif isinstance(score, tuple):
            category, sim = score
            raw_scores.append(category * 10.0 + sim)
            eligible.append(True)
        else:
            raw_scores.append(float(score))
            eligible.append(True)
    return np.asarray(raw_scores, dtype=float), np.asarray(eligible, dtype=bool)
