# OAKG benchmark definitions

Standalone, reviewable definition of the OAKG retrieval benchmark. The raw
phenotype **values** (organ/tumor morphometry from MSD Pancreas, LiTS, and
FLARE segmentation masks) are governed by the source dataset licences and are
**not** redistributed here; they regenerate from the licensed masks via
`python -m oakg.build_benchmark`. These files define the *task*:

- `queries.json` — 111 queries with primary/secondary predicates, source, split.
- `relevance.csv` — graded + binary relevance labels.
- `case_scopes.csv` — per-case source, split, native observed organs, split hash.
- `phenotype_schema.json` — feature → type + anatomical support (schema only).
- `anatomy.json` — organ vocabulary, per-source native scopes, support sets.
- `masking.json` — masking base seed and the full realization catalogue.

Regenerate: `python -m oakg.export_benchmark_defs --data data --out benchmark`
