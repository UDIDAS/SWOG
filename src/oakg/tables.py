"""Publication-ready result tables with bootstrap confidence intervals (Section 12)."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .stats import bootstrap_metric_ci


def format_metric_ci(mean: float, low: float, high: float, digits: int = 3) -> str:
    if not np.isfinite(mean):
        return "--"
    return f"{mean:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def publication_table(
    results: pd.DataFrame,
    group_columns: Sequence[str] = ("track", "masking_regime", "method"),
    metrics: Sequence[str] = ("P@10", "R@10", "AP", "nDCG@10"),
    n_bootstrap: int = 10_000,
    seed: int = 2027,
) -> pd.DataFrame:
    rows = []
    for keys, group in results.groupby(list(group_columns)):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, keys))
        for metric in metrics:
            mean, low, high = bootstrap_metric_ci(
                group[metric].to_numpy(), n_bootstrap=n_bootstrap, seed=seed
            )
            label = "mAP" if metric == "AP" else metric
            row[label] = format_metric_ci(mean, low, high)
        row["ServedRate"] = float(group["served"].mean())
        row["n_queries"] = int(group["query_id"].nunique())
        row["n_served"] = int(group["served"].sum())
        rows.append(row)
    return pd.DataFrame(rows)
