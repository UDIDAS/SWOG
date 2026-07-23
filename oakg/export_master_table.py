"""Export the compact master nDCG@10 table (to-do list item 2).

One table of final absolute nDCG@10 for the primary method + baselines across the
four masking regimes, so the paper can be revised without mixing runs/versions.

Rows: OAKG-Lexicographic, masked cosine, zero imputation, missingness indicators,
WL graph kernel (all from the MAIN pipeline, ref track); OAKG-Union (from the
observation-boundary ablation — the only place OAKG-Union exists).
Columns: uniform, random, asymmetric, dataset-style.

Run:  python -m oakg.export_master_table --out results/tables
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REGIMES = ["uniform", "random", "asymmetric", "dataset_style"]
# main-pipeline methods (retrieval_summary.csv, ref track) -> display name
MAIN = [
    ("OAKG-lexicographic [ref]", "OAKG-Lexicographic"),
    ("Masked cosine [ref]", "Masked cosine"),
    ("Zero imputation [ref]", "Zero imputation"),
    ("Missingness indicators [ref]", "Missingness indicators"),
    ("WL [ref]", "WL graph kernel"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/tables")
    args = ap.parse_args()
    rdir = Path(args.results)

    rs = pd.read_csv(rdir / "tables" / "retrieval_summary.csv")
    rs = rs[rs.track == "ref"]
    ua = pd.read_csv(rdir / "union_ablation" / "union_ablation_method_summary.csv")

    rows = []
    for raw, disp in MAIN:
        row = {"method": disp}
        for rg in REGIMES:
            sub = rs[(rs.masking_regime == rg) & (rs.method == raw)]
            row[rg] = round(float(sub["nDCG@10"].mean()), 3) if len(sub) else None
        row["source"] = "main pipeline (ref, lexicographic primary)"
        rows.append(row)
    # OAKG-Union — only exists in the observation-boundary ablation
    urow = {"method": "OAKG-Union"}
    for rg in REGIMES:
        sub = ua[(ua.regime == rg) & (ua.method == "OAKG-Union")]
        urow[rg] = round(float(sub["nDCG@10"].iloc[0]), 3) if len(sub) else None
    urow["source"] = "observation-boundary ablation (ablation-internal)"
    rows.append(urow)

    df = pd.DataFrame(rows)[["method", *REGIMES, "source"]]
    out = Path(args.out)
    df.to_csv(out / "master_nDCG_table.csv", index=False)

    # LaTeX
    hdr = "Method & Uniform & Random & Asymmetric & Dataset-style \\\\"
    body = "\n".join(
        f"{r.method} & " + " & ".join(f"{getattr(r, rg):.3f}" for rg in REGIMES) + " \\\\"
        for r in df.itertuples(index=False)
    )
    tex = ("\\begin{tabular}{lcccc}\n\\toprule\n" + hdr +
           "\n\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n")
    (out / "master_nDCG_table.tex").write_text(tex)

    # Markdown
    md = ["# Master nDCG@10 table (final, `incomparable_policy: bottom`, lexicographic primary)\n",
          "Absolute nDCG@10 by masking regime. Rows 1–5 are the MAIN pipeline (ref track);",
          "OAKG-Union is ablation-internal (the only place it exists). Regenerate:",
          "`python -m oakg.export_master_table`.\n",
          "| Method | Uniform | Random | Asymmetric | Dataset-style |",
          "|---|---|---|---|---|"]
    for r in df.itertuples(index=False):
        md.append(f"| {r.method} | " + " | ".join(f"{getattr(r, rg):.3f}" for rg in REGIMES) + " |")
    md.append(
        "\n> **All-111 convention:** every value is averaged over the same 111 queries "
        "(a query a method cannot serve scores 0). So rows **1–5 (main pipeline) ARE "
        "directly subtractable** — e.g. OAKG-Lexicographic − Masked cosine gives the "
        "paired effect.\n"
        "> **Exception — do NOT subtract the OAKG-Union row.** OAKG-Union comes from the "
        "separate observation-boundary ablation (its own accounting), not the main "
        "pipeline. The valid OAKG-vs-Union effect is the **paired Δ** in "
        "`results/union_ablation/` (asymmetric +0.068, dataset-style +0.112), not the "
        "difference of these two absolute rows.\n")
    md.append("Sources: rows 1–5 `results/tables/retrieval_summary.csv` (ref); "
              "OAKG-Union `results/union_ablation/union_ablation_method_summary.csv`.")
    (out / "master_nDCG_table.md").write_text("\n".join(md) + "\n")

    print("wrote master_nDCG_table.{csv,tex,md}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
