# OAKG retrieval demos

A Streamlit app: one heading, **three retrieval panels** (tabs), and a single **assistant** chatbot
at the bottom that can answer about any panel. (The assistant is a local Llama 3.2 3B, but the UI
just calls it "assistant".)

## Run
```bash
conda activate llmft            # env with streamlit + pandas + sklearn + transformers/torch
streamlit run app/oakg_query_app.py
```
Opens at http://localhost:8501. No prep — reads `kg/data/corpus_perpatient.json`
(512 patient-level KG instances: 281 Pancreas, 131 LiTS, 100 FLARE).

## 🔎 Tab 1 — Range query
Structured query: pick a phenotype and a condition. **Phenotypes are numeric (organ / tumor volumes)
OR categorical (tumor burden / multiplicity / containment / location)** — numeric ones take an
operator + threshold (`< / ≤ / > / ≥ / = / between`), categorical ones take a value set; an optional
extra categorical filter can be AND-ed on and a **competitor** to compare against OAKG. You can also **describe the query in natural language** (expander at the top): the assistant fills the controls **and narrows the dropdown options to that organ scope** (e.g. describe pancreatic tumors → phenotype/dataset/categorical options limit to the pancreas; **Show all options** resets). Paper queries live in their own tab.

- **OAKG** returns only **observation-backed** matches (a patient can only match a phenotype it
  actually observed — missing = unknown, never imputed).
- The **competitor** (Zero / Mean / Median imputation, or Cross-organ collision) fabricates a value
  for unobserved phenotypes and **over-returns** — false positives.
- Shows headline metrics, a **precision leaderboard** (OAKG vs every competitor), the two ranked
  panels side by side (⚠️ = false positive). Sharpest case: `= 0` ("no tumor")
  — the competitor sweeps in every unobserved patient; OAKG keeps only the truly-measured zeros.
- The retrieved patients are drawn as **one merged KG** (top-N by relevance): each patient's subgraph,
  all linked through **shared Dataset and SNOMED/NCIt concept nodes** — the shared schema makes them a
  single connected graph rather than N separate ones.

## 🧭 Tab 2 — Similar patients
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
high`) — plus companion bars.

## 📄 Tab 3 — Paper queries
The OAKG paper's queries — **run within this tab** (no loading into other tabs):
- **Structured (Table 4):** boolean phenotype queries (High burden, Multifocal, Tumor-in-organ,
  Contained, Small). Click one and it runs OAKG's observation-backed retrieval right here (OAKG
  matches vs coverage-blind, table, merged KG). *Cross-organ distribution* is flagged
  **indeterminate** (single-organ observability → OAKG returns "unknown").
- **Cross-dataset (B1–B7):** each is a plain-English sentence — *start from patient X (one dataset),
  find similar patients in another dataset that share an organ*. Click **Run** and it scores OAKG vs
  Masked cosine here. B3/B4/B7 use slice-level FLARE query patients absent from this patient-level
  demo, so they're disabled.

## ➕ Tab 4 — New CT → KG
Ingest a **new labeled CT**: upload the CT + its label mask (organs 1 liver, 2 R-kidney, 3 spleen,
4 pancreas, 13 L-kidney; tumor 14). SAM3 produces **accurate GT-box-prompted** predictions (liver
Dice ~0.97, pancreas ~0.87), we **validate** them (per-organ volume plausibility + Dice vs the mask),
show a prediction overlay, extract the KG phenotypes, and on **Add** the patient becomes queryable in
the other tabs + the assistant.

**A label mask is required** — and that is a real limitation, not a shortcut: our SAM3 models are
box-prompted, and without a mask to supply localization the segmentation isn't good enough for a
quality KG (autonomous tumor Dice ≈ 0.11; organs over-segment). The mask gives the box; SAM3 refines
it into an accurate prediction. (Backend: `ingest.py`, reusing `src/scripts/infer_sam3.py`.)

## 💬 Assistant (bottom of the page)
A single chatbot below the tabs, available regardless of which panel you're on. It's given the
**current state of all three panels** (queries + results), so it can *Summarize the current panels*
or answer follow-ups about any of them. Grounded in that context; the chat persists until you clear
it. (Backend: local Llama 3.2 3B; the UI just says "assistant".)

## The observability is real (not synthetic)
From each patient's `observed_organs` + which dataset annotated tumors: Pancreas patients imaged only
the pancreas, LiTS only the liver (24 with an observed-zero liver tumor), FLARE 5 organ volumes but
no tumors. That's why a pancreas patient and a FLARE patient share only `pancreas_volume`.

## Files
- `oakg_query_app.py` — the Streamlit app (three tabs + one bottom assistant; reusable `llm_block`).
- `paper_retrieval.py` — the OAKG paper's baselines + OAKG scorer, vendored verbatim (Streamlit-free).
- `kg_viz.py` — Plotly + vis.js KG visualization builders (Streamlit-free).
- `llm_backend.py` — Llama 3.2 3B load + generate helpers (Streamlit-free).
- `assets/vis-network.min.js` — vendored vis.js for the interactive graph.
