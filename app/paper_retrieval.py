#!/usr/bin/env python3
"""
OAKG paper baselines (Section 5) + OAKG scorer (Section 6), vendored so the demo's competitors
are IDENTICAL to the paper (oakg/baselines.py + oakg/oakg.py in the oakg branch).

Retrieval task = patient similarity: rank candidate patients against a query patient over a masked
phenotype matrix X (NaN where unobserved) with observation mask M. Baselines differ only in how they
treat unobserved features; OAKG restricts to jointly-observed features and weights by a shared-
evidence coefficient gamma (Jaccard of observed organ sets).

`build_corpus(records)` turns our patient-level KG instances into (Corpus, X, M).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Corpus:
    features: list
    feature_type: dict
    feature_means: np.ndarray
    feature_modes: np.ndarray
    feature_ranges: np.ndarray
    case_to_row: dict
    observation_sets: dict          # case_id -> frozenset(observed organs); drives OAKG's gamma


# ----------------------------------------------------------------- baselines (verbatim logic)
def cosine_pair(xq, xj):
    nq, nj = np.linalg.norm(xq), np.linalg.norm(xj)
    return 0.0 if nq == 0 or nj == 0 else float(np.dot(xq, xj) / (nq * nj))


def zero_imputed_similarity(q_idx, cand_idx, X, M, corpus):
    xq = np.nan_to_num(X[q_idx], nan=0.0)
    xc = np.nan_to_num(X[cand_idx], nan=0.0)
    return cosine_similarity(xq[None, :], xc)[0]


def mean_imputed_similarity(q_idx, cand_idx, X, M, corpus):
    fill = np.array([corpus.feature_means[j] if corpus.feature_type[f] == "numeric"
                     else corpus.feature_modes[j] for j, f in enumerate(corpus.features)])
    xq = np.where(M[q_idx], X[q_idx], fill)
    xc = np.where(M[cand_idx], X[cand_idx], fill)
    return cosine_similarity(xq[None, :], xc)[0]


def missing_indicator_similarity(q_idx, cand_idx, X, M, corpus):
    xq = np.nan_to_num(X[q_idx], nan=0.0)
    xc = np.nan_to_num(X[cand_idx], nan=0.0)
    zq = np.concatenate([xq, M[q_idx].astype(float)])
    zc = np.concatenate([xc, M[cand_idx].astype(float)], axis=1)
    return cosine_similarity(zq[None, :], zc)[0]


def masked_cosine_similarity(q_idx, cand_idx, X, M, corpus):
    """The key simple baseline: compare only jointly observed features."""
    scores = np.full(len(cand_idx), np.nan)
    for i, j in enumerate(cand_idx):
        joint = M[q_idx] & M[j]
        if np.any(joint):
            scores[i] = cosine_pair(X[q_idx, joint], X[j, joint])
    return scores


def gower_similarity(q_idx, cand_idx, X, M, corpus):
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


BASELINE_FUNCTIONS = {
    "Zero imputation": zero_imputed_similarity,
    "Mean imputation": mean_imputed_similarity,
    "Missingness indicators": missing_indicator_similarity,
    "Masked cosine": masked_cosine_similarity,
    "Gower": gower_similarity,
}

# whether each baseline fabricates a value for unobserved features (coverage-blind) or not
COVERAGE_BLIND = {"Zero imputation": True, "Mean imputation": True,
                  "Missingness indicators": True, "Masked cosine": False, "Gower": False}


# ----------------------------------------------------------------- OAKG (verbatim logic)
def observation_overlap(q_case, c_case, corpus):
    oq, oc = corpus.observation_sets[q_case], corpus.observation_sets[c_case]
    inter, union = oq & oc, oq | oc
    return inter, (len(inter) / len(union) if union else 0.0)


def oakg_similarity_components(q_idx, c_idx, X, M, corpus):
    joint = M[q_idx] & M[c_idx]
    if not np.any(joint):
        return None
    numeric = np.array([corpus.feature_type[f] == "numeric" for f in corpus.features])
    comps = []
    nj = joint & numeric
    if np.any(nj):
        d = np.abs(X[q_idx, nj] - X[c_idx, nj])
        comps.append(float(np.mean(np.maximum(0.0, 1.0 - d / corpus.feature_ranges[nj]))))
    sj = joint & ~numeric
    if np.any(sj):
        comps.append(float(np.mean(X[q_idx, sj] == X[c_idx, sj])))
    return float(np.mean(comps)) if comps else None


def oakg_scores(query_case, candidate_ids, X, M, corpus, policy="product", gamma_min=0.25):
    """Return (scores, eligible, gammas). policy 'product' = gamma * similarity (paper default)."""
    q_idx = corpus.case_to_row[query_case]
    scores, elig, gammas = [], [], []
    for cand in candidate_ids:
        inter, gamma = observation_overlap(query_case, cand, corpus)
        gammas.append(gamma)
        if not inter:
            scores.append(np.nan); elig.append(False); continue
        s = oakg_similarity_components(q_idx, corpus.case_to_row[cand], X, M, corpus)
        if s is None or (policy == "threshold" and gamma < gamma_min):
            scores.append(np.nan); elig.append(False); continue
        scores.append(gamma * s if policy == "product" else s)
        elig.append(True)
    return np.asarray(scores, float), np.asarray(elig, bool), np.asarray(gammas, float)


# ----------------------------------------------------------------- build corpus from our KG
# (feature name, type, kind, organ) — observability follows organ imaging + tumor annotation
FEATURES = [
    ("liver_volume", "numeric", "organ", "liver"),
    ("pancreas_volume", "numeric", "organ", "pancreas"),
    ("spleen_volume", "numeric", "organ", "spleen"),
    ("right_kidney_volume", "numeric", "organ", "right_kidney"),
    ("left_kidney_volume", "numeric", "organ", "left_kidney"),
    ("pancreas_tumor_volume", "numeric", "tumor", "pancreas"),
    ("liver_tumor_volume", "numeric", "tumor", "liver"),
]
_TUMOR_ANNOTATED = {("pancreas", "pancreas"), ("liver", "lits")}


def _observed(rec, kind, organ):
    if organ not in rec["observed_organs"]:
        return False
    return True if kind == "organ" else (organ, rec["dataset"]) in _TUMOR_ANNOTATED


def build_corpus(records):
    n, d = len(records), len(FEATURES)
    X, M = np.full((n, d), np.nan), np.zeros((n, d), bool)
    obs_sets, case_to_row = {}, {}
    for i, rec in enumerate(records):
        case_to_row[rec["case_id"]] = i
        obs_sets[rec["case_id"]] = frozenset(rec["observed_organs"])
        for j, (_, _, kind, organ) in enumerate(FEATURES):
            if _observed(rec, kind, organ):
                od = rec["organs"].get(organ, {})
                v = od.get("organ_volume_cm3") if kind == "organ" else od.get("tumor_volume_cm3")
                if v is not None:
                    X[i, j], M[i, j] = v, True
    masked = np.where(M, X, np.nan)
    means = np.nan_to_num(np.nanmean(masked, axis=0))
    ranges = np.nanmax(masked, axis=0) - np.nanmin(masked, axis=0)
    ranges[~np.isfinite(ranges) | (ranges == 0)] = 1.0
    feats = [f[0] for f in FEATURES]
    corpus = Corpus(feats, {f[0]: f[1] for f in FEATURES}, means, means.copy(), ranges,
                    case_to_row, obs_sets)
    return corpus, X, M


def rank(query_case, method, corpus, X, M, records, k=10):
    """Rank all other patients for `method` ('OAKG' or a baseline name). Returns list of dicts."""
    case_ids = [r["case_id"] for r in records]
    ds = {r["case_id"]: r["dataset"] for r in records}
    cand = [c for c in case_ids if c != query_case]
    cand_idx = np.array([corpus.case_to_row[c] for c in cand])
    q_obs = corpus.observation_sets[query_case]
    if method == "OAKG":
        scores, elig, gammas = oakg_scores(query_case, cand, X, M, corpus)
    else:
        scores = BASELINE_FUNCTIONS[method](corpus.case_to_row[query_case], cand_idx, X, M, corpus)
        elig = ~np.isnan(scores)
        gammas = np.array([len(q_obs & corpus.observation_sets[c]) / len(q_obs | corpus.observation_sets[c])
                           if (q_obs | corpus.observation_sets[c]) else 0.0 for c in cand])
    order = np.argsort(-np.nan_to_num(scores, nan=-1e18))
    out = []
    for o in order[:k]:
        shared = sorted(q_obs & corpus.observation_sets[cand[o]])
        out.append({"patient": cand[o], "dataset": ds[cand[o]],
                    "score": (None if np.isnan(scores[o]) else round(float(scores[o]), 3)),
                    "shared organs": ", ".join(shared) or "—",
                    "γ": round(float(gammas[o]), 2),
                    "low evidence (γ<0.25)": bool(gammas[o] < 0.25)})
    return out
