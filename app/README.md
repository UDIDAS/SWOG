# OAKG phenotype-query demo

A minimal Streamlit app that retrieves patients from the imaging KG by **phenotype + desirable
range**, run two ways side by side to show why **observability-awareness (OAKG)** is more reliable
than a coverage-blind retriever that imputes missing values to 0. Includes a **Llama 3.2 3B**
explainer/chatbot to discuss the results in plain language.

## Run
```bash
conda activate llmft            # env with streamlit + pandas + transformers/torch
streamlit run app/oakg_query_app.py
```
Opens at http://localhost:8501. No data prep — it reads `kg/data/corpus_perpatient.json`
(512 patient-level KG instances: 281 Pancreas, 131 LiTS, 100 FLARE).

## Controls
- **Datasets** — restrict the cohort (default: all three).
- **Numeric phenotype + range slider** — the desirable range to retrieve on.
- **Optional categorical filter** — burden / multiplicity / location.
- **Rows to show per panel** — 5 → 200 (default 10).

## The two panels
| Panel | Missing value = | Effect |
|---|---|---|
| ❌ **Coverage-blind** (non-OAKG) | **0** (imputed) | Patients never measured for the phenotype get 0 and *falsely* match "small/low" ranges. |
| ✅ **OAKG** (observability-aware) | **unobserved / unknown** | Only patients that actually observed the phenotype are returned. |

Both panels are ranked by the **same relevance** (closeness to the range midpoint), so the genuinely
observed patients appear in the **same relative order** in both — the coverage-blind panel just
injects extra ⚠️ imputed-zero rows among them. The headline metric reports how many coverage-blind
hits are **false matches**, and the *"who OAKG keeps vs drops"* box shows, for the current query, one
real patient OAKG keeps next to one it drops — the two are indistinguishable to coverage-blind.

## Llama 3.2 3B explainer + chatbot
- **📝 Explain these results** — generates a plain-language summary of the current query grounded in
  the retrieval facts.
- **Chat box** — ask follow-ups ("why did coverage-blind return more?", "which datasets caused the
  false matches?"). The model is fed a structured context (dataset-level observability, false-match
  counts by dataset, example patients) and is instructed to ground only in those facts.
- Model: `meta-llama/Llama-3.2-3B-Instruct` (bf16 on GPU), loaded once and cached. First use loads
  ~6 GB of weights; needs a CUDA GPU (tested on an L40S, ~1–2 s per reply). Falls back to the ungated
  `unsloth/Llama-3.2-3B-Instruct` mirror if the gated repo is unavailable.

## The observability is real (not synthetic)
Derived from each patient's `observed_organs` + which dataset annotated tumors:
- **Pancreas** patients imaged only the pancreas → liver/spleen/kidney UNOBSERVED.
- **LiTS** patients imaged only the liver; 24 have an **observed-zero** liver tumor (imaged, none found).
- **FLARE** patients imaged 5 organ volumes but tumors were never annotated (organs-only) → their
  stored `tumor_volume = 0.0` is a **false zero**.

Example — query *Liver tumor volume ∈ [0, 5] cm³* over all 512 patients:
coverage-blind returns **503** (381 false: the 281 Pancreas + 100 FLARE patients that never imaged a
liver tumor, imputed to 0), while OAKG returns **122** patients that genuinely observed a small/zero
liver tumor.

## Files
- `oakg_query_app.py` — the Streamlit app.
- `llm_backend.py` — Llama 3.2 3B load + generate helpers (Streamlit-free, so it's reusable/testable).
