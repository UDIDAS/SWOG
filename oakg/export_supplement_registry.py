"""Emit the concrete values the AAAI supplement placeholders need (final-checkup §3).

Fills the [[FILL]] items with verified repository values: the 17-variable
phenotype/observability registry + support sets + normalization ranges, the
component/weight/threshold/ranking definitions, masking seeds, predicted-mask
cohort coverage, the P@10/R@10/mAP table, and the hardware/software environment.

Run:  python -m oakg.export_supplement_registry --data data --out results/audit/supplement_registry.md
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, TOP_K
from .data import load_experiment_data, Corpus
from .oakg import rank_policy  # noqa: F401  (documents the policy signature)
from .phenotypes import MIN_LESION_VOXELS

GAMMA_MIN = 0.25          # eligibility threshold (eta) — oakg.rank_policy default
EVIDENCE_BINS = (0.25, 0.50, 0.75)   # lexicographic gamma-category boundaries
BBOX_MARGIN = 2           # phenotypes.bbox_contained default


def _env() -> list[str]:
    mods = {}
    for m in ["numpy", "pandas", "scipy", "sklearn", "networkx", "statsmodels", "torch"]:
        try:
            mods[m] = __import__(m).__version__
        except Exception:
            mods[m] = "not installed"
    cpu = mem = "n/a"
    try:
        import os
        cpu = f"{os.cpu_count()} logical cores"
        with open("/proc/meminfo") as f:
            kb = int(next(l for l in f if l.startswith("MemTotal")).split()[1])
        mem = f"{kb // 1024 // 1024} GB"
    except Exception:
        pass
    return [
        f"- OS: {platform.platform()}",
        f"- CPU: {cpu}",
        f"- Memory: {mem}",
        "- GPU: not required for the retrieval pipeline (CPU-only); optional for the "
        "CompGCN backbone and upstream segmentation.",
        f"- Python: {sys.version.split()[0]}",
        "- Libraries: " + ", ".join(f"{k} {v}" for k, v in mods.items()),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/audit/supplement_registry.md")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    cfg = Config(use_demo_data=False, data_dir=args.data)
    data = load_experiment_data(cfg.data_dir)
    corpus = Corpus.build(data, cfg)

    # 17-variable registry: feature, type, support organs, normalization range.
    support = {}
    for r in pd.read_csv(Path(args.data) / "phenotypes_ref.csv")[
            ["feature", "support_organs"]].drop_duplicates("feature").itertuples(index=False):
        support[r.feature] = "" if pd.isna(r.support_organs) else str(r.support_organs)
    reg_rows = []
    for i, f in enumerate(corpus.features):
        typ = corpus.feature_type[f]
        rng = float(corpus.feature_ranges[i]) if typ == "numeric" else ""
        sup = support.get(f, "") or "(global)"
        reg_rows.append(f"| `{f}` | {typ} | {sup} | {rng if rng == '' else round(rng, 4)} |")

    L = []
    L.append("# OAKG supplement registry (verified values)\n")
    L.append("Concrete values for the AAAI supplement `[[FILL]]` placeholders, generated "
             "from the frozen benchmark. Regenerate: "
             "`python -m oakg.export_supplement_registry`.\n")

    L.append("## 1. Phenotype + observability registry (17 variables)\n")
    L.append("| Feature | Type | Support organs | Normalization range |")
    L.append("|---|---|---|---|")
    L += reg_rows
    L.append(f"\nNormalization: numeric features scaled by the per-feature range above "
             f"(min–max over the corpus); component similarity uses "
             f"`max(0, 1 − |Δ|/range)`. Top-k = {TOP_K}.\n")

    L.append("## 2. Component definitions and weights\n")
    L.append("- **numeric** component: mean over jointly-observed numeric features of "
             "`max(0, 1 − |x_q − x_c| / range_f)`.")
    L.append("- **categorical_relational** component: mean exact-match over jointly-observed "
             "set-like / binary features.")
    L.append("- **Weights: uniform.** The pair similarity is the unweighted mean of the "
             "present component group(s) — no learned or per-feature weights "
             "(`oakg.oakg.oakg_similarity_components`).\n")

    L.append("## 3. Thresholds and ranking parameters\n")
    L.append(f"- Lesion threshold: a tumor connected component must exceed "
             f"**{MIN_LESION_VOXELS} voxels** to count (`MIN_LESION_VOXELS`).")
    L.append(f"- Tumor-in-organ containment bbox margin: **±{BBOX_MARGIN} voxels**.")
    L.append(f"- Eligibility threshold η (γ_min, threshold policy): **{GAMMA_MIN}**.")
    L.append(f"- Lexicographic evidence-bin boundaries: **{list(EVIDENCE_BINS)}** "
             f"(γ-category = number of boundaries met, 0–3).")
    L.append(f"- Bootstrap B: **{cfg.n_bootstrap}** paired resamples; Holm-corrected; seed "
             f"**{cfg.seed}**.")
    L.append("- γ (shared-evidence coefficient) = Jaccard overlap of the two cases' "
             "observed-organ sets.\n")

    L.append("## 4. Masking seeds and realizations\n")
    seed = cfg.seed
    L.append(f"- Base seed: **{seed}**.")
    L.append(f"- uniform: seed {seed} (no additional masking).")
    L.append(f"- random: fractions 0.20/0.40/0.60/0.80, seeds {seed+1}–{seed+4}.")
    L.append(f"- dataset-style: pancreas_only / liver_only / kidney_only / multi_organ, seed {seed}.")
    L.append(f"- asymmetric: broad_to_narrow + narrow_to_broad, seed {seed}.")
    L.append("- Full machine-readable catalogue: `benchmark/masking.json`.\n")

    L.append("## 5. Predicted-mask cohort coverage\n")
    cases = data.cases
    L.append("Reference (GT) track covers all cases; the predicted track is partial:")
    L.append("- Pancreas / LiTS: SSL/SAM3 predictions paired with GT (pred track).")
    L.append("- FLARE: predictions for **20 of 100** cases (`sam3_delivery`); the other 80 "
             "use GT on both tracks.")
    L.append(f"- Cases by source: " + ", ".join(
        f"{k}={v}" for k, v in cases.dataset.value_counts().to_dict().items()) + ".\n")

    L.append("## 6. Precision@10 / Recall@10 / mAP (primary lexicographic, ref)\n")
    pt = pd.read_csv(Path(args.results) / "tables" / "publication_ready_retrieval_table.csv")
    sub = pt[(pt.track == "ref") & (pt.method == "OAKG-lexicographic [ref]")][
        ["masking_regime", "P@10", "R@10", "mAP", "nDCG@10"]]
    L.append("| Regime | P@10 | R@10 | mAP | nDCG@10 |")
    L.append("|---|---|---|---|---|")
    for regime, p, rec, mapv, ndcg in sub.itertuples(index=False, name=None):
        L.append(f"| {regime} | {p} | {rec} | {mapv} | {ndcg} |")
    L.append("\nFull table (all methods/regimes/tracks): "
             "`results/tables/publication_ready_retrieval_table.csv`.\n")

    L.append("## 7. Ontology concepts and identifiers (KG-schema grounding)\n")
    L.append("The MMKG schema grounds entities and phenotype values in standard medical "
             "terminologies. The T-Box is `benchmark/kg_schema.owl`; the concept→code "
             "alignment is `benchmark/ontology_mappings.json` (our curated mapping, which "
             "references standard codes rather than redistributing the source terminologies).\n")
    ont_path = Path("benchmark/ontology_mappings.json")
    if ont_path.exists():
        om = json.loads(ont_path.read_text())
        st = om.get("stats", {})
        L.append(f"Coverage: {st.get('concepts', len(om.get('mappings', {})))} concepts "
                 f"({', '.join(f'{k} {v}' for k, v in st.get('by_system', {}).items() if v)}).\n")
        L.append("| Entity | System:Code (display) |")
        L.append("|---|---|")
        for ent, codes in om.get("mappings", {}).items():
            cs = " · ".join(f"{sys}:{i['code']} ({i['display']})" for sys, i in codes.items())
            L.append(f"| `{ent}` | {cs or '—'} |")
        L.append("")

    L.append("## 8. Hardware and software environment\n")
    L += _env()
    L.append("\n- Runtime: the full paper pipeline (`make reproduce-paper`, 10k bootstrap) "
             "completes in a few minutes on the above CPU; storage for tracked results "
             "is < 5 MB (raw masks/embeddings excluded).")

    Path(args.out).write_text("\n".join(L) + "\n")
    print(f"wrote {args.out}  ({len(corpus.features)} features, "
          f"{len(data.cases)} cases)")


if __name__ == "__main__":
    main()
