"""Paired bootstrap CIs, Holm correction, policy selection, upstream degradation."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


def paired_bootstrap_difference(
    query_level: pd.DataFrame,
    method_a: str,
    method_b: str,
    metric: str = "nDCG@10",
    n_bootstrap: int = 10_000,
    seed: int = 2027,
) -> dict[str, float]:
    """Paired bootstrap over queries for ``metric(method_a) - metric(method_b)``."""
    pivot = query_level.pivot_table(index="query_id", columns="method", values=metric)
    pair = pivot[[method_a, method_b]].dropna()
    if len(pair) == 0:
        return {"n": 0, "difference": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan}

    diffs = (pair[method_a] - pair[method_b]).to_numpy()
    local_rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sample = local_rng.choice(diffs, size=len(diffs), replace=True)
        boot[i] = np.mean(sample)

    estimate = float(np.mean(diffs))
    ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
    p = 2.0 * min(np.mean(boot <= 0), np.mean(boot >= 0))
    return {
        "n": len(diffs),
        "difference": estimate,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p": float(min(1.0, p)),
    }


def comparison_family(
    query_level: pd.DataFrame,
    target_method: str,
    baselines: Sequence[str],
    metric: str = "nDCG@10",
    n_bootstrap: int = 10_000,
    seed: int = 2027,
) -> pd.DataFrame:
    """Compare ``target_method`` against each baseline with Holm correction."""
    rows = []
    for baseline in baselines:
        result = paired_bootstrap_difference(
            query_level, target_method, baseline, metric=metric,
            n_bootstrap=n_bootstrap, seed=seed,
        )
        result.update({"target": target_method, "baseline": baseline, "metric": metric})
        rows.append(result)
    out = pd.DataFrame(rows)
    valid = out["p"].notna()
    if valid.any():
        out.loc[valid, "p_holm"] = multipletests(out.loc[valid, "p"], method="holm")[1]
    return out


def bootstrap_metric_ci(
    values: np.ndarray,
    n_bootstrap: int = 10_000,
    seed: int = 2027,
) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    local_rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        boot[i] = np.mean(local_rng.choice(values, len(values), replace=True))
    return float(values.mean()), *map(float, np.quantile(boot, [0.025, 0.975]))


def upstream_degradation(summary: pd.DataFrame) -> pd.DataFrame:
    """Delta_upstream = Metric_ref - Metric_pred, per method/metric/masking (Section 3)."""
    tmp = summary.copy()
    tmp["method_base"] = tmp["method"].str.replace(r" \[(ref|pred)\]$", "", regex=True)
    pivot = tmp.pivot_table(
        index=["masking_regime", "mask_metadata", "method_base"],
        columns="track",
        values=["P@10", "R@10", "mAP", "nDCG@10", "ServedRate"],
    )
    rows = []
    for idx, row in pivot.iterrows():
        result = {"masking_regime": idx[0], "mask_metadata": idx[1], "method": idx[2]}
        for metric in ["P@10", "R@10", "mAP", "nDCG@10", "ServedRate"]:
            if (metric, "ref") in row.index and (metric, "pred") in row.index:
                result[f"Delta_upstream_{metric}"] = row[(metric, "ref")] - row[(metric, "pred")]
        rows.append(result)
    return pd.DataFrame(rows)


def choose_policy(
    query_results: pd.DataFrame,
    track: str = "ref",
    metric: str = "nDCG@10",
) -> pd.DataFrame:
    """Rank OAKG policies by validation ``metric`` then served rate (Section 6)."""
    policy_rows = query_results[
        query_results["track"].eq(track)
        & query_results["method"].str.startswith("OAKG-")
    ].copy()
    return (
        policy_rows.groupby("method")
        .agg(
            mean_metric=(metric, "mean"),
            served_rate=("served", "mean"),
            n_queries=("query_id", "nunique"),
        )
        .reset_index()
        .sort_values(["mean_metric", "served_rate"], ascending=False)
    )
