"""Observation masks and the four masking regimes.

Each query and candidate carries its own observation mask; the exact retained
organ set is recorded in every realization's metadata for reproducibility.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import Config
from .data import Corpus, parse_organs


@dataclass(frozen=True)
class MaskingRealization:
    name: str
    seed: int
    observation_sets: dict[str, frozenset[str]]
    metadata: dict[str, Any]


def make_uniform_mask(
    cases: pd.DataFrame,
    retained_organs: Sequence[str],
    seed: int,
) -> MaskingRealization:
    retained = frozenset(retained_organs)
    obs = {
        row.case_id: parse_organs(row.available_organs) & retained
        for row in cases.itertuples(index=False)
    }
    return MaskingRealization("uniform", seed, obs, {"retained_organs": sorted(retained)})


def make_random_mask(
    cases: pd.DataFrame,
    missing_fraction: float,
    seed: int,
) -> MaskingRealization:
    local_rng = np.random.default_rng(seed)
    obs = {}
    for row in cases.itertuples(index=False):
        available = sorted(parse_organs(row.available_organs))
        if not available:
            obs[row.case_id] = frozenset()
            continue
        keep_n = max(1, int(round(len(available) * (1.0 - missing_fraction))))
        kept = local_rng.choice(available, size=min(keep_n, len(available)), replace=False)
        obs[row.case_id] = frozenset(map(str, kept))
    return MaskingRealization(
        "random", seed, obs, {"missing_fraction": float(missing_fraction)}
    )


def make_dataset_style_mask(
    cases: pd.DataFrame,
    style: str,
    organs: Sequence[str],
    seed: int,
) -> MaskingRealization:
    styles = {
        "pancreas_only": {"pancreas"},
        "liver_only": {"liver"},
        "kidney_only": {"left_kidney", "right_kidney"},
        "multi_organ": set(organs),
    }
    if style not in styles:
        raise ValueError(f"Unknown style: {style}")
    target = styles[style]
    obs = {
        row.case_id: parse_organs(row.available_organs) & target
        for row in cases.itertuples(index=False)
    }
    return MaskingRealization("dataset_style", seed, obs, {"style": style})


def make_asymmetric_mask(
    cases: pd.DataFrame,
    query_case_ids: set[str],
    direction: str,
    organs: Sequence[str],
    seed: int,
    narrow: Sequence[str] = ("liver",),
) -> MaskingRealization:
    if direction not in {"broad_to_narrow", "narrow_to_broad"}:
        raise ValueError(direction)
    broad = set(organs)
    narrow_set = set(narrow)
    obs = {}
    for row in cases.itertuples(index=False):
        available = parse_organs(row.available_organs)
        is_query = row.case_id in query_case_ids
        target = broad if (is_query == (direction == "broad_to_narrow")) else narrow_set
        obs[row.case_id] = available & target
    return MaskingRealization("asymmetric", seed, obs, {"direction": direction})


def feature_mask_from_observation_sets(
    observation_sets: Mapping[str, frozenset[str]],
    corpus: Corpus,
) -> np.ndarray:
    mask = np.zeros((len(corpus.case_order), len(corpus.features)), dtype=bool)
    for i, cid in enumerate(corpus.case_order):
        observed = observation_sets[cid]
        for j, feature in enumerate(corpus.features):
            support = corpus.feature_support[feature]
            mask[i, j] = support.issubset(observed)
    return mask


def apply_mask(
    x_full: np.ndarray,
    base_available: np.ndarray,
    realization: MaskingRealization,
    corpus: Corpus,
) -> tuple[np.ndarray, np.ndarray]:
    support_mask = feature_mask_from_observation_sets(realization.observation_sets, corpus)
    observed = base_available & support_mask
    X = x_full.copy()
    X[~observed] = np.nan
    return X, observed


def build_default_realizations(
    data_cases: pd.DataFrame,
    corpus: Corpus,
    config: Config,
) -> list[MaskingRealization]:
    """The full set of masking realizations required by the study (Section 4)."""
    seed = config.seed
    organs = list(config.organs)
    return [
        make_uniform_mask(data_cases, organs, seed=seed),
        *[
            make_random_mask(data_cases, frac, seed=seed + i)
            for i, frac in enumerate([0.20, 0.40, 0.60, 0.80], start=1)
        ],
        *[
            make_dataset_style_mask(data_cases, style, organs, seed=seed)
            for style in ["pancreas_only", "liver_only", "kidney_only", "multi_organ"]
        ],
        make_asymmetric_mask(
            data_cases, corpus.query_case_ids, "broad_to_narrow", organs, seed=seed
        ),
        make_asymmetric_mask(
            data_cases, corpus.query_case_ids, "narrow_to_broad", organs, seed=seed
        ),
    ]
