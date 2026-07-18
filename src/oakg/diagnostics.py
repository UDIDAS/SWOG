"""Query and candidate-pool diagnostics (Section 11.1)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .data import ExperimentData


def query_diagnostics(data: ExperimentData) -> tuple[pd.DataFrame, pd.DataFrame]:
    query_rows = []
    for query in data.queries.itertuples(index=False):
        primary = json.loads(query.primary_predicates)
        secondary = json.loads(query.secondary_predicates)
        rel = data.relevance[data.relevance["query_id"].eq(query.query_id)]
        n_rel = int(rel["binary_relevance"].sum())
        query_rows.append({
            "query_id": query.query_id,
            "query_case_id": query.query_case_id,
            "primary_features": "|".join(p["feature"] for p in primary),
            "secondary_features": "|".join(p["feature"] for p in secondary),
            "n_primary": len(primary),
            "n_secondary": len(secondary),
            "candidate_pool_size": len(rel),
            "n_relevant": n_rel,
            "relevance_prevalence": n_rel / len(rel) if len(rel) else np.nan,
            "zero_relevant": n_rel == 0,
        })
    query_level = pd.DataFrame(query_rows)

    summary = pd.DataFrame({
        "n_queries": [len(query_level)],
        "mean_candidate_pool": [query_level["candidate_pool_size"].mean()],
        "median_candidate_pool": [query_level["candidate_pool_size"].median()],
        "mean_relevant": [query_level["n_relevant"].mean()],
        "median_relevant": [query_level["n_relevant"].median()],
        "mean_prevalence": [query_level["relevance_prevalence"].mean()],
        "zero_relevant_queries": [query_level["zero_relevant"].sum()],
    })
    return query_level, summary
