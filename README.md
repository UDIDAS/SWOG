# SWOG — Autonomous Abdominal-CT Segmentation → Imaging Knowledge Graph

**Aim.** Given a **new abdominal CT with no annotations**, automatically segment **liver, kidney, and
pancreas + their tumors**, turn them into an **ontology-grounded knowledge graph**, and use that KG to
retrieve similar patients, validate new cases, and grow over time — a knowledge base that keeps
improving as patients arrive.

---

## Why the current method — how we got here (results-driven)

**1. The blocker: our models needed ground truth to run.**
We began by reproducing AUSAM (SAM1 + prompts derived from the GT mask). It gave strong Dice
(pancreas organ ~0.83, tumor ~0.91), but **every prediction required a ground-truth box/point** — so it
could not touch a *new, unlabeled* scan. That defeats the whole "new patient → KG" goal.

**2. Decision → switch the backbone to SAM3.** SAM3 adds **concept/text prompting** (segment `"liver"`
with no box). On a held-out CT this gave **liver Dice 0.964 autonomous vs 0.974 semi-oracle** — a
0.01 gap. *Result: organ segmentation without any label is real.*

**3. Concept prompting fails on tumors (0.37).** Foundation models don't know medical tumors. *Decision
→ train one generic tumor model* (prompt `"tumor"`, no box) on **pooled** abdominal-CT tumors. *Result:
0.37 → 0.93* as we pooled LiTS → +Pancreas → +FLARE → +KiTS.

**4. It doesn't transfer to unseen tumor types (measured: 0.02 cross-dataset).** The liver+pancreas
model scored **0.02** on unseen kidney tumors. *So coverage must be trained in* — which is exactly why
the pool grows, and each dataset lifts it. Not a caveat; the design principle.

**5. Scope → liver / kidney / pancreas.** These are the three organs with matching tumor datasets
(LiTS→liver, KiTS→kidney, Pancreas→pancreas), with **FLARE** for cross-dataset diversity. Organs get
the *same* recipe as tumors (a generic organ model). Everything is tested on **held-out test sets with
no GT**.

**6. The KG is a living knowledge base, not a dump.** RDF/OWL, **direct triples**
(`lesion —tumorBurden→ "high"`), grounded to **SNOMED/LOINC/ICD/MeSH** via a live mapper (no hard-coded
codes), open-world. For a new (GT-free) patient it **validates** each phenotype against the cohort
distribution and **retrieves** similar cases — and every admitted patient sharpens the next validation.

---

## The pipeline today

```
new CT (no labels)
  → organs:  SAM3 concept prompt  ("liver"/"kidney"/"pancreas")      [autonomous]
  → tumor:   generic tumor model  (prompt "tumor", trained)          [autonomous]
  → masks → phenotypes (volume, diameter, centroid, burden, …)
  → ontology-grounded KG  →  retrieval · GT-free validation · growth
```
Segmentation: `infer_ensemble.py`. KG build: `build_flare23_enriched_kg.py` + `kg_grounding.py`.
Interactive KG: `app/oakg_query_app.py`. Walkthroughs: `notebooks/`.

## Segmentation results (held-out, honest)

**Autonomous — the deployment numbers (no ground truth, concept prompts):**

| Structure | Autonomous (concept) | Semi-oracle ceiling (GT box) | Note |
|---|---|---|---|
| Liver organ | **0.964** | 0.974 | ≈ no cost to dropping the label |
| Pancreas organ | 0.642 | 0.75–0.88 | weaker; generic organ model in progress |
| Kidney / spleen | *sweep in progress* | 0.96 | large & distinct — expected strong |
| **Tumor** (generic model, v3) | **~0.93** | 0.91 (per-organ box) | training closes the gap |
| Tumor, *unseen* type (cross-dataset) | 0.02 | — | must train the type in |

Tumor trajectory as coverage grew: **0.37 → 0.909 → 0.9145 → 0.93** (base concept → LiTS+Pancreas →
+FLARE → +KiTS). Generic tumor pool ≈ 20.7k slices; generic organ pool ≈ 16k (liver/kidney/pancreas).

**Semi-oracle backbone (fine-tuned SAM3, GT box) — per-dataset ceilings.** Case-level = whole patients
held out (the honest generalization number); slice-level leaks patient features and reads higher.

| Dataset | Split | Organ Dice | Tumor Dice |
|---|---|---|---|
| Pancreas (MSD Task07) | case-level | 0.866 | 0.894 |
| LiTS (liver tumor) | case-level | — | 0.840 |
| FLARE (13 organs + tumor) | slice-level | liver 0.973 · pancreas 0.907 · duodenum 0.913 | 0.883 |

## The knowledge graph — how it's used and how it evolves

Each patient is a subgraph: `ImagingCase → Organ / Lesion`, with phenotypes as **direct triples**
(`gt_volume_cm3`, `gt_max_diameter_mm`, `gt_centroid_mm`, `tumorBurden`, `lesionMultiplicity`,
`lesion_count`), every entity grounded to **SNOMED / LOINC / ICD / MeSH** via a live mapper — no
hard-coded codes. Current graph: FLARE23, **1,312 patients, 13 organs + tumor** (`kg/graph/`).

**Used as a knowledge base, not a store:**
- **Retrieval** — SPARQL ("largest kidney tumors", "tumors per organ") **and observability-aware
  similarity (OAKG)**: patients are compared over *jointly-observed* phenotypes, weighted by shared
  evidence (γ), so an unobserved organ reads as *unknown*, **never 0**.
- **Semantic interoperability** — ontology grounding lets external hierarchies reason over it
  (e.g. a kidney tumor *is-a* genitourinary neoplasm).
- **GT-free validation of a new patient** — each autonomously-segmented phenotype is scored against the
  cohort distribution (percentile / z-score); an implausible value (e.g. a 4,900 cc "liver") is flagged
  as a segmentation error — no ground truth required.

**How it evolves with each incoming patient:**
```
new CT → autonomous segmentation → phenotypes
       → GT-free plausibility check against the KG        (admit / flag)
       → admitted patient joins the global query KG
       → cohort distributions sharpen → the NEXT patient is validated better
```
It is **open-world**: a new patient may carry a phenotype never seen before — the mapper grounds the new
term and it joins the graph with **no schema migration**. The graph that stores the data is the same one
that **interprets and validates** new, unlabeled scans, and it improves as it grows.
Interactive: `app/oakg_query_app.py` · runnable demo: `notebooks/KG_as_Knowledge_Base.ipynb`.

## Repository

```
notebooks/      Segmentation_Results · KG_as_Knowledge_Base · Test_KG_from_CT   (runnable)
app/            oakg_query_app.py — interactive KG retrieval/validation
kg/             schema.owl · ontology_mappings.json (grounding cache) · graph/ (gitignored outputs)
src/scripts/    segmentation (train_tumor_generic, build_*_pool, extract_*, infer_ensemble),
                KG (kg_grounding, build_flare23_enriched_kg, flare23_predict, kg_build_*),
                AUSAM/SAM3 backbones (run_flare, run_*_sam3, infer_sam3)
archive/        superseded scripts/notebooks/docs (local; gitignored)
```

## Environment
```
conda env: llmft  ·  Python 3.11, PyTorch 2.5.1+cu121  ·  2× NVIDIA L40S (48 GB)
torch · transformers (SAM3) · monai · rdflib · scikit-image · scipy · nibabel
```
