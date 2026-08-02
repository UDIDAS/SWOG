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
Interactive KG: `app/oakg_query_app.py`. Walkthroughs: `src/notebooks/`.

## Segmentation results (held-out, honest)

**How to read these.** *Dice* = overlap between the predicted and ground-truth mask (0–1; higher is
better, ≈0.9+ is strong). Each result depends on **how the model is prompted**:

- **Autonomous (concept prompt)** — the model gets *only the image + a word* (e.g. `"liver"`), no
  annotation. **This is the real deployment number** — what runs on a new, unlabeled scan.
- **Semi-oracle ceiling (GT box)** — the model is *handed a bounding box drawn from the ground-truth
  mask* (told *where* the structure is). It's an **upper bound**: needs a label, so it can't run on new
  data — it only shows how much a label would be worth.
- The **gap** between the two columns = the cost of working without annotations.

**Autonomous vs. ceiling, per structure:**

| Structure | Autonomous *(concept, no label)* | Semi-oracle ceiling *(given GT box)* | Gap → what it means |
|---|:--:|:--:|---|
| Liver organ | **0.964** | 0.974 | 0.01 — the label barely helps |
| Pancreas organ | 0.642 | 0.75–0.88 | ~0.10 — weaker; organ model in progress |
| Kidney / spleen | *sweep pending* | ~0.96 | large & distinct — expect strong |
| **Tumor** (generic model) | **0.938** | 0.91 *(per-organ box)* | training *beats* the per-organ box |
| Tumor, *unseen* type | 0.02 | — | fails until that type is in training |

*Tumor Dice as training coverage grew:* **0.37 → 0.909 → 0.9145 → 0.938**
(base concept → LiTS+Pancreas → +FLARE → +KiTS). Pools ≈ 20.7k tumor slices, ≈ 16k organ slices.

**Semi-oracle backbone ceilings, per dataset** (fine-tuned SAM3 given the GT box). Here *Split* = how
train/test were divided, which decides whether the number is honest:

- **case-level** = *whole patients* held out → honest (predicts performance on a brand-new patient).
- **slice-level** = random 2D slices → a patient's near-identical slices can land in *both* train and
  test (data leakage → inflated). Only used where the dataset has no patient IDs.

| Dataset | Split *(how test was held out)* | Organ Dice | Tumor Dice |
|---|---|:--:|:--:|
| Pancreas (MSD Task07) | case-level *(whole patients)* | 0.866 | 0.894 |
| LiTS (liver tumor) | case-level *(whole patients)* | — *(liver copied from GT)* | 0.840 |
| FLARE (13 organs + tumor) | slice-level *(no patient IDs)* | liver 0.973 · pancreas 0.907 · duodenum 0.913 | 0.883 |

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
Interactive: `app/oakg_query_app.py` · runnable demo: `src/notebooks/KG_as_Knowledge_Base.ipynb`.

## Research direction & contribution

**Thesis.** The novelty is *not* the segmentation — that builds on foundation models (SAM3) and is on
par with, not ahead of, SOTA (K-Prism, GF-Screen). It is the **downstream reasoning layer**: an
**observability-aware, self-evolving clinical knowledge graph (OAKG)** built end-to-end from
*label-free* autonomous segmentation.

**The problem we own.** Unifying many imaging datasets into one global KG creates **structural
partial-observability** — each source annotated different organs (Pancreas→pancreas, LiTS→liver,
FLARE→5 organs). Standard retrieval either **imputes** the missing values (→ false matches) or does
**naive masked similarity** (→ a "perfect match" on a single shared feature). As the field unifies ever
more datasets (PanTS, K-Prism), this problem *grows* — and no one is addressing it.

**Contribution, precisely.**
1. **OAKG** — *evidence-calibrated* retrieval/reasoning under structural partial-observability:
   unobserved is never imputed, and matches are weighted by joint observability (**γ**), so answers
   stay correct as the graph scales. Eliminates the false positives/negatives that imputation and
   masked-similarity produce on a merged multi-source KG.
2. **A GT-free, self-evolving pipeline** — autonomous segmentation → ontology-grounded KG that
   validates each new *unlabeled* patient against the cohort and *improves as it grows*.
3. **Autonomous CT→KG as infrastructure** (SAM3 concept organs + generic tumor model), positioned on
   SOTA segmentation rather than competing with it.

**Experimental program.**
- **A. OAKG (method) — first result in.** Structured multi-source benchmark: patients are drawn from
  single-site "datasets" with **disjoint observed organs** (Pancreas→pancreas, LiTS→liver, KiTS→kidneys)
  plus a full-observation FLARE hub — mirroring the real merge. In the realistic mixed regime OAKG gives
  the **best retrieval (Precision@10 0.78 vs 0.73 mean-impute)** and the **fewest spurious matches (0.14
  vs 0.17)**. The **γ ablation is decisive**: masked-cosine *without* γ collapses to **0.08 / 0.84**
  (a single shared organ reads as a perfect match); γ recovers it to **0.78 / 0.14**. At extreme
  disjointness (no shared evidence) no method can retrieve — OAKG returns the honest floor instead of
  fabricating a signal, unlike imputation. Script: `src/scripts/exp_oakg_structured.py`.
- **B. Framework — B1 + B2 results in.**
  - **B1 (predicted-KG vs GT-KG fidelity).** Does the KG built from *autonomous* masks answer like the
    GT-KG? On 20 paired pancreas cases (mean Dice **0.919**) it matches almost exactly: volume MAPE
    **4.8%** (r=0.992), diameter MAPE 0.9% (r=0.997), centroid within **1.4 mm**, **95%** size-bin
    agreement, and the "rank by pancreas size" query has **Spearman 0.986 / top-3 overlap 1.0**. Takeaway:
    at good segmentation the predicted KG is faithful — **segmentation quality, not KG construction, is
    the bottleneck**. Script: `src/scripts/exp_kg_fidelity.py`.
  - **B2 (self-evolving GT-free validation).** Plant realistic segmentation errors (a volume leak that
    breaks the volume↔diameter relation) and score plausibility with **no ground truth**. The KG's
    **joint** model (cohort covariance over 5 organs × {log-volume, log-diameter}) separates clean-vs-error
    at **AUROC 0.78 vs 0.66** for a marginal per-organ range check (**+0.12** — it sees inconsistencies the
    marginal check can't), and it **improves as the KG grows** (0.74→0.79 over N=15→120) then plateaus once
    the 10-D covariance is well-estimated; the marginal baseline stays flat. Richer phenotypes (13 organs +
    tumor) would extend the curve. Script: `src/scripts/exp_oakg_evolve.py`.
- **C. Segmentation:** autonomous per-structure Dice, cross-dataset generalization, vs the semi-oracle
  ceiling — compared against K-Prism / GF-Screen / PanTS.

**Target venues.** NeurIPS Datasets & Benchmarks (benchmark framing) or ICLR/AAAI (OAKG-as-method);
MICCAI / health-AI as strong domain fits.

## Repository

```
app/            oakg_query_app.py — interactive KG retrieval/validation
kg/             schema.owl · ontology_mappings.json (grounding cache) · graph/ (gitignored outputs)
src/
  notebooks/    Segmentation_Results · KG_as_Knowledge_Base · Test_KG_from_CT   (runnable)
  scripts/      segmentation (train_tumor_generic, build_*_pool, extract_*, infer_ensemble),
                KG (kg_grounding, build_flare23_enriched_kg, flare23_predict, kg_build_*),
                AUSAM/SAM3 backbones (run_flare, run_*_sam3, infer_sam3)
```

## Environment
```
conda env: llmft  ·  Python 3.11, PyTorch 2.5.1+cu121  ·  2× NVIDIA L40S (48 GB)
torch · transformers (SAM3) · monai · rdflib · scikit-image · scipy · nibabel
```
