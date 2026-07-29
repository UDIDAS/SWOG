# OAKG retrieval demos

A Streamlit app with **two separate retrievals**, one per tab, plus a Llama 3.2 3B explainer in each.

## Run
```bash
conda activate llmft            # env with streamlit + pandas + sklearn + transformers/torch
streamlit run app/oakg_query_app.py
```
Opens at http://localhost:8501. No prep — reads `kg/data/corpus_perpatient.json`
(512 patient-level KG instances: 281 Pancreas, 131 LiTS, 100 FLARE).

## 🔎 Tab 1 — Range-based retrieval
Structured query: pick a phenotype and a condition (`< / ≤ / > / ≥ / = / between` + threshold,
optional categorical filter) and a **competitor** to compare against OAKG.

- **OAKG** returns only **observation-backed** matches (a patient can only match a phenotype it
  actually observed — missing = unknown, never imputed).
- The **competitor** (Zero / Mean / Median imputation, or Cross-organ collision) fabricates a value
  for unobserved phenotypes and **over-returns** — false positives.
- Shows headline metrics, a **precision leaderboard** (OAKG vs every competitor), the two ranked
  panels side by side (⚠️ = false positive), and a Llama explainer. Sharpest case: `= 0` ("no tumor")
  — the competitor sweeps in every unobserved patient; OAKG keeps only the truly-measured zeros.
- The retrieved patients are drawn as **one merged KG** (top-N by relevance): each patient's subgraph,
  all linked through **shared Dataset and SNOMED/NCIt concept nodes** — the shared schema makes them a
  single connected graph rather than N separate ones.

## 🧭 Tab 2 — Anchor-based similarity
Pick **any patient** as the anchor; rank all others by similarity using the **OAKG paper's exact
baselines** (Section 5), vendored verbatim from `oakg/baselines.py` + `oakg/oakg.py`.

| Method | Missing features | Coverage |
|---|---|---|
| **Zero / Mean imputation** | fabricated constant, cosine | coverage-blind |
| **Missingness indicators** | cosine on `[features, mask]` | partly aware |
| **Masked cosine** | cosine over jointly-observed features | coverage-aware |
| **Gower** | Gower over jointly-observed features | coverage-aware |
| **OAKG** | jointly-observed **+ γ** (shared-evidence Jaccard weight) | the proposed guardrail |

Metric: **weak-overlap neighbours (γ<0.25) in the top-k** — lower is better. The paper's finding
shows directly: for a pancreas anchor, **Masked cosine** ranks FLARE patients as *perfect* matches
(they share only `pancreas_volume`, so a one-feature cosine is trivially 1.0), while **OAKG**
down-weights them via γ. Includes an **interactive vis.js KG view with a toggle** — *Merged KG*
(anchor + its OAKG neighbours as one graph, linked through shared Dataset/concept nodes) or *Single
patient* (the anchor's own subgraph; categorical phenotypes are direct triples `lesion —tumorBurden→
high`) — plus companion bars and a Llama explainer.

## 🗣 Tab 3 — Describe / paper queries
A natural-language front-end plus the OAKG paper's queries, highlighted.

- **Paper queries (click to load):**
  - *Structured (Table 4):* High tumor burden, Multifocal disease, Tumor in an organ, Contained
    tumor, Small tumor. (*Cross-organ distribution* is called out as **indeterminate** under
    single-organ observability — OAKG returns "unknown" instead of a false answer, the paper's point.)
  - *Cross-dataset complex (B1–B7):* the paper's cross-dataset queries; each button sends its query
    patient to the 🧭 Anchor tab (they're the similarity paradigm).
- **Natural language:** type a description ("small pancreatic tumors that are contained"); Llama 3.2
  3B maps it to a structured query and fills the panel (a robust normalizer fixes the 3B model's
  sloppy keys). Edit the phenotype / condition / categorical panel, then it runs OAKG retrieval
  inline — OAKG matches vs what coverage-blind (impute 0) would return, a results table, and the
  retrieved patients drawn as **one merged KG**.

## The observability is real (not synthetic)
From each patient's `observed_organs` + which dataset annotated tumors: Pancreas patients imaged only
the pancreas, LiTS only the liver (24 with an observed-zero liver tumor), FLARE 5 organ volumes but
no tumors. That's why a pancreas patient and a FLARE patient share only `pancreas_volume`.

## Files
- `oakg_query_app.py` — the Streamlit app (two tabs; a reusable `llm_block` form-based chat).
- `paper_retrieval.py` — the OAKG paper's baselines + OAKG scorer, vendored verbatim (Streamlit-free).
- `kg_viz.py` — Plotly + vis.js KG visualization builders (Streamlit-free).
- `llm_backend.py` — Llama 3.2 3B load + generate helpers (Streamlit-free).
- `assets/vis-network.min.js` — vendored vis.js for the interactive graph.
