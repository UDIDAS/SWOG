"""Export the OAKG benchmark *definitions* as standalone, reviewable files.

The raw phenotype VALUES (organ/tumor morphometry derived from MSD/LiTS/FLARE
segmentation masks) are governed by the source dataset licences and are NOT
redistributed here — they regenerate from the licensed masks via
``python -m oakg.build_benchmark``. What this module exports is the benchmark
*structure*, which is reviewable on its own:

  benchmark/queries.json        111 queries with primary/secondary predicates,
                                source, and split.
  benchmark/relevance.csv       graded + binary relevance labels.
  benchmark/case_scopes.csv     per-case source, split, native observed organs,
                                and a deterministic split hash.
  benchmark/phenotype_schema.json   feature -> {feature_type, support_organs}
                                (schema only; no values).
  benchmark/anatomy.json        organ vocabulary, per-source native observation
                                scopes, and anatomical support sets.
  benchmark/masking.json        masking seeds and the full realization catalogue.
  benchmark/README.md           describes every file + the licensing note.

Run:  python -m oakg.export_benchmark_defs --data data --out benchmark
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .config import Config, ORGANS

SOURCE = {"Pancreas": "msd_pancreas", "LiTS": "lits", "FLARE": "flare22"}


def _split_hash(case_id: str) -> str:
    return hashlib.md5(case_id.encode()).hexdigest()[:12]


def main() -> None:
    ap = argparse.ArgumentParser(description="Export standalone benchmark definitions.")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="benchmark")
    args = ap.parse_args()
    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cases = pd.read_csv(data / "cases.csv")
    queries = pd.read_csv(data / "queries.csv")
    relevance = pd.read_csv(data / "relevance.csv")
    pheno = pd.read_csv(data / "phenotypes_ref.csv")

    src_of = dict(zip(cases.case_id, cases.dataset.map(SOURCE)))
    split_of = dict(zip(cases.case_id, cases.split))

    # 1. queries.json (predicates parsed back to objects) --------------------
    qdefs = []
    for q in queries.itertuples(index=False):
        qdefs.append({
            "query_id": q.query_id,
            "query_case_id": q.query_case_id,
            "source": src_of.get(q.query_case_id),
            "split": split_of.get(q.query_case_id),
            "primary_predicates": json.loads(q.primary_predicates),
            "secondary_predicates": json.loads(q.secondary_predicates),
        })
    (out / "queries.json").write_text(json.dumps(qdefs, indent=1) + "\n")

    # 2. relevance.csv (labels only) -----------------------------------------
    relevance[["query_id", "candidate_id", "binary_relevance", "graded_relevance"]] \
        .to_csv(out / "relevance.csv", index=False)

    # 3. case_scopes.csv ------------------------------------------------------
    scopes = cases.assign(
        source_id=cases.dataset.map(SOURCE),
        observed_organs=cases.available_organs,
        split_hash=cases.case_id.map(_split_hash),
    )[["case_id", "source_id", "split", "observed_organs", "split_hash"]]
    scopes.to_csv(out / "case_scopes.csv", index=False)

    # 4. phenotype_schema.json (feature -> type + support; NO values) --------
    schema = (pheno[["feature", "feature_type", "support_organs"]]
              .drop_duplicates("feature").fillna(""))
    schema_map = {r.feature: {"feature_type": r.feature_type,
                              "support_organs": [o for o in str(r.support_organs).split("|") if o]}
                  for r in schema.itertuples(index=False)}
    (out / "phenotype_schema.json").write_text(json.dumps(schema_map, indent=1) + "\n")

    # 5. anatomy.json — vocabulary + per-source native scopes + support sets --
    per_source_scope = {}
    for ds, sid in SOURCE.items():
        organs = set()
        for r in cases[cases.dataset == ds].itertuples(index=False):
            organs |= {o for o in str(r.available_organs).split("|") if o}
        per_source_scope[sid] = sorted(organs)
    anatomy = {
        "organ_vocabulary": list(ORGANS),
        "native_observation_scopes": per_source_scope,
        "support_sets": {f: v["support_organs"] for f, v in schema_map.items()},
    }
    (out / "anatomy.json").write_text(json.dumps(anatomy, indent=1) + "\n")

    # 6. masking.json — seeds + realization catalogue -------------------------
    seed = Config().seed
    masking = {
        "base_seed": seed,
        "organ_vocabulary": list(ORGANS),
        "realizations": [
            {"regime": "uniform", "seed": seed, "note": "no additional masking (native scopes)"},
            *[{"regime": "random", "missing_fraction": f, "seed": seed + i}
              for i, f in enumerate([0.20, 0.40, 0.60, 0.80], start=1)],
            *[{"regime": "dataset_style", "style": s, "seed": seed}
              for s in ["pancreas_only", "liver_only", "kidney_only", "multi_organ"]],
            {"regime": "asymmetric", "direction": "broad_to_narrow", "seed": seed},
            {"regime": "asymmetric", "direction": "narrow_to_broad", "seed": seed},
        ],
    }
    (out / "masking.json").write_text(json.dumps(masking, indent=1) + "\n")

    # 7. README ---------------------------------------------------------------
    (out / "README.md").write_text(
        "# OAKG benchmark definitions\n\n"
        "Standalone, reviewable definition of the OAKG retrieval benchmark. The raw\n"
        "phenotype **values** (organ/tumor morphometry from MSD Pancreas, LiTS, and\n"
        "FLARE segmentation masks) are governed by the source dataset licences and are\n"
        "**not** redistributed here; they regenerate from the licensed masks via\n"
        "`python -m oakg.build_benchmark`. These files define the *task*:\n\n"
        f"- `queries.json` — {len(qdefs)} queries with primary/secondary predicates, source, split.\n"
        "- `relevance.csv` — graded + binary relevance labels.\n"
        "- `case_scopes.csv` — per-case source, split, native observed organs, split hash.\n"
        "- `phenotype_schema.json` — feature → type + anatomical support (schema only).\n"
        "- `anatomy.json` — organ vocabulary, per-source native scopes, support sets.\n"
        "- `masking.json` — masking base seed and the full realization catalogue.\n\n"
        "Regenerate: `python -m oakg.export_benchmark_defs --data data --out benchmark`\n"
    )

    print(f"wrote {out}/  queries={len(qdefs)} relevance_rows={len(relevance)} "
          f"cases={len(cases)} features={len(schema_map)} "
          f"realizations={len(masking['realizations'])}")


if __name__ == "__main__":
    main()
