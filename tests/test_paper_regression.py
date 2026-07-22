"""Numerical regression test: guard the frozen paper numbers against drift.

Every value below is pinned to the committed 10k-bootstrap paper results. If a
regeneration changes a headline number, this test fails and forces a review
before the manuscript / snapshot go out of sync. Run: python -m pytest tests/ -q
"""
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
TOL = 1e-4


def _val(df, value_col, **filters):
    for c, v in filters.items():
        df = df[df[c].astype(str) == str(v)]
    assert len(df) == 1, f"expected 1 row, got {len(df)} for {filters}"
    return float(df.iloc[0][value_col])


# ---- union-ablation observation-boundary effect (primary lexicographic) ----
def test_union_ablation_paired_deltas():
    pc = pd.read_csv(ROOT / "results/union_ablation/union_ablation_paired_comparison.csv")
    assert _val(pc, "delta_obs", regime="uniform") == pytest.approx(0.000769, abs=TOL)
    assert _val(pc, "delta_obs", regime="random") == pytest.approx(-0.009730, abs=TOL)
    assert _val(pc, "delta_obs", regime="asymmetric") == pytest.approx(0.068095, abs=TOL)
    assert _val(pc, "delta_obs", regime="dataset_style") == pytest.approx(0.111879, abs=TOL)


def test_union_ablation_random_oakg_absolute():
    ms = pd.read_csv(ROOT / "results/union_ablation/union_ablation_method_summary.csv")
    # ablation-internal value (all queries, incomparable@bottom) — see README note.
    assert _val(ms, "nDCG@10", regime="random", method="OAKG") == pytest.approx(0.402694, abs=TOL)


# ---- native cross-dataset (primary lexicographic) --------------------------
def test_native_primary_effects():
    ms = pd.read_csv(ROOT / "results/native_cross_dataset/native_method_summary.csv")
    assert _val(ms, "mean", stratum="cross-source-pooled", method="OAKG") == pytest.approx(0.312497, abs=TOL)
    assert _val(ms, "mean", stratum="cross-source-pooled", method="OAKG-Union") == pytest.approx(0.142862, abs=TOL)
    assert _val(ms, "mean", stratum="same-source", method="OAKG") == pytest.approx(0.399087, abs=TOL)
    # strong baselines beat OAKG on raw cross-source (documented caveat)
    assert _val(ms, "mean", stratum="cross-source-pooled", method="MissingnessIndicators") == pytest.approx(0.625961, abs=TOL)
    assert _val(ms, "mean", stratum="cross-source-pooled", method="ZeroImputation") == pytest.approx(0.584759, abs=TOL)


def test_native_policy_invariance():
    """Native result is exactly policy-invariant: lexicographic == similarity."""
    lex = pd.read_csv(ROOT / "results/native_cross_dataset/native_method_summary.csv")
    sim = pd.read_csv(ROOT / "results/native_cross_dataset/similarity_policy/native_method_summary.csv")
    m = lex.merge(sim, on=["family", "stratum", "method"], suffixes=("_lex", "_sim"))
    assert (m.mean_lex - m.mean_sim).abs().max() == pytest.approx(0.0, abs=1e-9)


# ---- hard-distractor CI (matches the manuscript [0.029, 0.113]) ------------
def test_hard_distractor_ci():
    hd = pd.read_csv(ROOT / "results/strata/hard_distractor.csv")
    row = hd[hd.method == "OAKG"].iloc[0]
    assert float(row.ci_low) == pytest.approx(0.0286, abs=TOL)
    assert float(row.ci_high) == pytest.approx(0.1127, abs=TOL)


# ---- ranking-policy tie: lexicographic == product on validation ------------
def test_policy_selection_tie():
    ps = pd.read_csv(ROOT / "results/tables/ranking_policy_selection.csv")
    lex = _val(ps, "mean_metric", method="OAKG-lexicographic [ref]")
    prod = _val(ps, "mean_metric", method="OAKG-product [ref]")
    assert lex == pytest.approx(prod, abs=1e-9)


# ---- cross-file invariant: unmasked OAKG is one number everywhere ----------
def test_unmasked_oakg_consistent_across_files():
    ua = pd.read_csv(ROOT / "results/union_ablation/union_ablation_method_summary.csv")
    ms = pd.read_csv(ROOT / "results/native_cross_dataset/native_method_summary.csv")
    u = _val(ua, "nDCG@10", regime="uniform", method="OAKG")
    n = _val(ms, "mean", stratum="same-source", method="OAKG")
    assert u == pytest.approx(n, abs=1e-9)
