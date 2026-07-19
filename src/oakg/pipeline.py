"""End-to-end driver: regenerate every deliverable table/figure from a Config.

Run with ``python -m oakg.pipeline`` (uses demo data) or point ``Config`` at a
real benchmark. Every output file listed in Section 13 of the guidelines is
written under ``config.results_dir``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .benchmark import run_full_benchmark, summarize
from .config import Config
from .consistency import ranking_consistency
from .data import Corpus, ExperimentData, load_data, validate_data
from .diagnostics import query_diagnostics
from .masking import apply_mask, build_default_realizations, make_random_mask, make_uniform_mask
from .neural import cross_backbone_table
from .selective import selective_curve
from .semantics import structured_query_evaluation
from .stats import choose_policy, upstream_degradation
from .tables import publication_table


@dataclass
class PipelineOutputs:
    data: ExperimentData
    corpus: Corpus
    results: pd.DataFrame
    summary: pd.DataFrame
    policy_selection: pd.DataFrame


def run_all(config: Config | None = None) -> PipelineOutputs:
    config = config or Config()
    config.ensure_dirs()
    qdir = config.query_level_dir
    tdir = config.tables_dir
    fdir = config.figures_dir

    # Archive the exact configuration behind these outputs.
    (config.results_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2))

    data = load_data(config)
    validate_data(data)
    corpus = Corpus.build(data, config)
    realizations = build_default_realizations(data.cases, corpus, config)

    # Sections 8/9 (Tables A/B): full benchmark + summaries.
    results = run_full_benchmark(realizations, data, corpus, config)
    results.to_csv(qdir / "query_level_retrieval_results.csv", index=False)
    summary = summarize(results)
    summary.to_csv(tdir / "retrieval_summary.csv", index=False)

    # Section 3: reference-to-predicted upstream degradation.
    upstream_degradation(summary).to_csv(tdir / "upstream_degradation.csv", index=False)

    # Section 6 (Table C): policy selection on the random-masking family.
    policy_selection = choose_policy(
        results[results["masking_regime"].eq("random")], track="ref"
    )
    policy_selection.to_csv(tdir / "ranking_policy_selection.csv", index=False)

    # Section 9 (Table D): cross-backbone observability (Base vs +Obs).
    cross_backbone_table(results).to_csv(tdir / "cross_backbone_observability.csv", index=False)

    # Section 7 (Figure 1): risk-coverage curve.
    curve_real = make_random_mask(data.cases, 0.60, config.seed + 60)
    cx, cm = apply_mask(corpus.x_ref_full, corpus.m_ref_full, curve_real, corpus)
    rc = selective_curve(cx, cm, curve_real, data, corpus, config)
    rc.to_csv(tdir / "risk_coverage_curve.csv", index=False)
    _plot_risk_coverage(rc, fdir / "risk_coverage_curve.png")

    # Section 10.1 (Table E): full-to-partial ranking consistency.
    full_real = make_uniform_mask(data.cases, config.organs, config.seed)
    part_real = make_random_mask(data.cases, 0.60, config.seed + 101)
    consistency = pd.concat([
        _consistency(corpus.x_ref_full, corpus.m_ref_full, full_real, part_real, "ref", data, corpus),
        _consistency(corpus.x_pred_full, corpus.m_pred_full, full_real, part_real, "pred", data, corpus),
    ], ignore_index=True)
    consistency.to_csv(tdir / "ranking_consistency.csv", index=False)

    # Section 10.2 (Table F): structured-query semantics.
    sem_real = make_random_mask(data.cases, 0.60, config.seed + 130)
    sx, sm = apply_mask(corpus.x_ref_full, corpus.m_ref_full, sem_real, corpus)
    detail, sem_summary = structured_query_evaluation(sx, sm, "ref", data, corpus)
    detail.to_csv(qdir / "structured_query_predictions.csv", index=False)
    sem_summary.to_csv(tdir / "structured_query_summary.csv", index=False)

    # Section 11.1: query diagnostics.
    diag, _ = query_diagnostics(data)
    diag.to_csv(tdir / "query_diagnostics.csv", index=False)

    # Section 12: publication-ready table with bootstrap CIs.
    publication_table(results, n_bootstrap=config.n_bootstrap, seed=config.seed).to_csv(
        tdir / "publication_ready_retrieval_table.csv", index=False
    )

    return PipelineOutputs(data, corpus, results, summary, policy_selection)


def _consistency(x_full, m_full, full_real, part_real, track, data, corpus):
    fx, fm = apply_mask(x_full, m_full, full_real, corpus)
    px, pm = apply_mask(x_full, m_full, part_real, corpus)
    return ranking_consistency(fx, fm, px, pm, full_real, part_real, track, data, corpus)


def _plot_risk_coverage(rc: pd.DataFrame, path) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(rc["coverage"], rc["risk"], marker="o")
    plt.xlabel("Served-query coverage")
    plt.ylabel("Selective risk = 1 - nDCG@10")
    plt.title("OAKG Retrieval Risk-Coverage")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


if __name__ == "__main__":
    out = run_all()
    print("Wrote deliverables to:", out.results.shape[0], "query-level rows")
    print(out.summary.sort_values("nDCG@10", ascending=False).head(10).to_string(index=False))
