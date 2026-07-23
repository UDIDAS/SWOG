"""Render every final paper table into one browsable file: results/PAPER_TABLES.md.

Reads the committed result CSVs and emits one document with all final tables,
grouped PRIMARY / SUPPLEMENTARY / EXPLORATORY, each with its source file. This is
the single place to see every table the paper uses.

Run:  python -m oakg.export_paper_tables --out results/PAPER_TABLES.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def render(df: pd.DataFrame, cols=None, rounder=3) -> str:
    if cols:
        df = df[cols]
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.{rounder}f}"
        return str(v)
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/PAPER_TABLES.md")
    args = ap.parse_args()
    R = Path(args.results)
    L = ["# OAKG — all final paper tables (one place)\n",
         "Generated from the committed result CSVs (corrected pipeline: "
         "`incomparable_policy: bottom`, OAKG-Lexicographic primary; snapshot "
         "`paper_snapshot.json`). Regenerate: `python -m oakg.export_paper_tables`.\n",
         "> **All-111 convention:** every value is averaged over the same 111 queries "
         "(a query a method cannot serve scores 0). Main-pipeline absolute columns are "
         "therefore directly comparable/subtractable. The one exception is **OAKG-Union**, "
         "which is ablation-internal (its own accounting) — for OAKG-vs-Union use the "
         "**paired Δ** in the union-ablation table, not a column subtraction.\n",
         "---\n## PRIMARY\n"]

    L.append("### Master nDCG@10 (all methods × regimes) — `tables/master_nDCG_table.csv`\n")
    L.append(render(pd.read_csv(R / "tables" / "master_nDCG_table.csv").drop(columns=["source"])))

    L.append("\n\n### Ranking-policy selection (Table C, validation) — `tables/ranking_policy_selection.csv`\n")
    L.append(render(pd.read_csv(R / "tables" / "ranking_policy_selection.csv"), rounder=4))

    L.append("\n\n### Strong baselines, OAKG-Lexicographic (ref), with CIs (Table A) — "
             "`tables/publication_ready_retrieval_table.csv`\n")
    pt = pd.read_csv(R / "tables" / "publication_ready_retrieval_table.csv")
    pt = pt[(pt.track == "ref") & (pt.method == "OAKG-lexicographic [ref]")][
        ["masking_regime", "P@10", "R@10", "mAP", "nDCG@10", "ServedRate"]]
    L.append(render(pt))
    L.append("\n(full baselines/tracks in the CSV.)")

    L.append("\n\n### Structured-query semantics (Table F) — `tables/structured_query_summary.csv`\n")
    sem = pd.read_csv(R / "tables" / "structured_query_summary.csv")
    L.append(render(sem[["semantics", "recall_F", "macro_F1", "indeterminate_rate",
                         "unsupported_negative_rate"]], rounder=3))
    L.append("\n`recall_F` = supported-negative recall. OAKG reaches unsupported-negative "
             "rate 0.000 (vs closed-world 0.014) at the cost of supported-negative recall "
             "and a 0.43 indeterminate rate.")

    L.append("\n\n---\n## SUPPLEMENTARY\n")
    L.append("### Observation-boundary ablation — OAKG vs OAKG-Union (paired) — "
             "`union_ablation/union_ablation_paired_comparison.csv`\n")
    ua = pd.read_csv(R / "union_ablation" / "union_ablation_paired_comparison.csv")
    L.append(render(ua[["regime", "OAKG", "OAKG_Union", "delta_obs", "ci_low", "ci_high", "holm_p"]]))

    L.append("\n\n### Native cross-dataset — OAKG vs OAKG-Union by source direction — "
             "`native_cross_dataset/native_paired_effects.csv`\n")
    ne = pd.read_csv(R / "native_cross_dataset" / "native_paired_effects.csv")
    ne = ne[(ne.family == "primary") & (ne.method_b == "OAKG-Union")][
        ["stratum", "n_queries", "delta", "ci_low", "ci_high", "p_value", "holm_p"]]
    L.append(render(ne))

    L.append("\n\n### Patient-level hard-distractor (paired vs zero-imp) — `strata/hard_distractor.csv`\n")
    L.append(render(pd.read_csv(R / "strata" / "hard_distractor.csv"), rounder=4))

    L.append("\n\n---\n## EXPLORATORY (not confirmatory)\n")
    L.append("### FLARE cross-organ tumor — **slice-level** — `strata/flare_tumor_realgt.csv`\n")
    L.append(render(pd.read_csv(R / "strata" / "flare_tumor_realgt.csv"), rounder=4))
    L.append("\n**Exploratory only:** slice-level; 364 slices ≠ 364 patients; patient "
             "clustering unreconstructable; CIs/p not treated as confirmatory.\n")

    Path(args.out).write_text("\n".join(L) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
