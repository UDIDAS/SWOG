# OAKG similarity-retrieval demo

A Streamlit app that demonstrates the OAKG paper's patient-similarity retrieval. Everything is driven
from the **sidebar**; the main page is the visualization.

## Run
```bash
conda activate llmft            # env with streamlit + pandas + sklearn + transformers/torch
streamlit run app/oakg_query_app.py
```
Opens at http://localhost:8501. No prep — it reads `kg/data/corpus_perpatient.json`
(512 patient-level KG instances: 281 Pancreas, 131 LiTS, 100 FLARE).

## Sidebar (all controls)
1. **Cohort** — which datasets are in play.
2. **Anchor patient** — a phenotype range query (`< / ≤ / > / ≥ / = / between` + threshold, optional
   categorical filter). OAKG returns only **observation-backed** matches; you pick one as the
   **anchor**. (This is where the observability guard lives: a patient can only match a phenotype it
   actually observed — no imputed zeros.)
3. **Similarity comparison** — which OAKG-paper baseline to pit against OAKG, and top-k.

## Main page (follows the anchor)

### 🔬 Similarity retrieval — OAKG vs the paper's baselines
The exact competitors from the OAKG paper (Section 5), vendored verbatim from `oakg/baselines.py` +
`oakg/oakg.py`. Task: rank every other patient against the anchor over a masked phenotype matrix.

| Method | Missing features | Coverage |
|---|---|---|
| **Zero imputation** | filled with 0, cosine | coverage-blind |
| **Mean imputation** | filled with mean/mode, cosine | coverage-blind |
| **Missingness indicators** | cosine on `[features, mask]` | partly aware |
| **Masked cosine** | cosine over **jointly-observed** features only | coverage-aware |
| **Gower** | Gower over jointly-observed features | coverage-aware |
| **OAKG** | jointly-observed **+ γ** (shared-evidence Jaccard weight) | the proposed guardrail |

The metric is **weak-overlap neighbours (γ<0.25) in the top-k** — lower is better. The paper's finding
shows directly: for a pancreas anchor, **Masked cosine** ranks FLARE patients as *perfect* matches
(they share only `pancreas_volume`, so a one-feature cosine is trivially 1.0), while **OAKG**
down-weights them via γ (they share 1/5 organs, γ=0.2) and returns true pancreas neighbours. A
per-baseline table scores all of them at once.

### 🕸 Knowledge graph — the anchor patient
Interactive **vis.js** graph of the anchor's KG subgraph (drag nodes, hover for properties, zoom):
`Patient → ImagingCase → Organ(s) → Lesion(s)` with categorical phenotypes as **direct triples**
(`lesion —tumorBurden→ high`), anatomic sites, and **SNOMED/NCIt concept** nodes. Companion bars show
the anchor's organ/tumor volumes and the datasets of its OAKG neighbours. Self-contained (vis-network
JS vendored in `assets/`, inlined — offline, no CDN).

### 🧠 Llama 3.2 3B explainer + chatbot
Explains the anchor's OAKG-vs-baseline retrieval in plain language and answers follow-ups, grounded
in a structured context (anchor, γ values, weak-overlap counts, top neighbours per method).
`meta-llama/Llama-3.2-3B-Instruct` (bf16 on GPU), loaded once. **The chat resets when you change the
anchor or comparison.**

## The observability is real (not synthetic)
From each patient's `observed_organs` + which dataset annotated tumors:
- **Pancreas** patients imaged only the pancreas → liver/spleen/kidney UNOBSERVED.
- **LiTS** patients imaged only the liver; 24 have an observed-zero liver tumor.
- **FLARE** patients imaged 5 organ volumes but tumors were never annotated (organs-only).

That's why a pancreas anchor and a FLARE patient share only `pancreas_volume` — the crux of the
masked-cosine-vs-OAKG contrast.

## Files
- `oakg_query_app.py` — the Streamlit app.
- `paper_retrieval.py` — the OAKG paper's baselines + OAKG scorer, vendored verbatim (Streamlit-free).
- `kg_viz.py` — Plotly + vis.js KG visualization builders (Streamlit-free).
- `llm_backend.py` — Llama 3.2 3B load + generate helpers (Streamlit-free).
- `assets/vis-network.min.js` — vendored vis.js for the interactive graph.
