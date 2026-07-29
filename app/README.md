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
- **Numeric phenotype + condition** — `< / ≤ / > / ≥ / = / between` with a threshold. A condition
  whose satisfying set includes 0 (`< X`, `≤ X`, `= 0`) exposes the imputation problem automatically,
  no slider fiddling. `= 0` is the sharpest case ("patients with *no* tumor").
- **Competitor (vs OAKG)** — choose which false-positive-raising baseline to pit against the OAKG
  guardrail: **zero / mean / median imputation** (fabricate a constant for missing values) or
  **cross-organ collision** (an untyped `tumor_volume` query is answered by *any* organ's value —
  right number, wrong organ). Each fabrication produces false positives on a different set of queries.
- **Optional categorical filter** — burden / multiplicity / location.
- **Rows to show per panel** — 5 → 200 (default 10).

## Retrieval-quality leaderboard
Above the panels, a leaderboard scores **OAKG against every competitor at once** on the current
query: `returned`, `false positives`, and **precision** (= fraction of a method's returns that are
observation-backed, i.e. agree with OAKG). OAKG is the reference at 1.00; each competitor drops below
1.00 exactly when its fabricated value satisfies the condition. This is the direct
"OAKG guardrail vs all false-positive cases" comparison.

## The two panels
| Panel | Missing value = | Effect |
|---|---|---|
| ❌ **Selected competitor** | **fabricated** (imputed constant, or borrowed from another organ) | Patients never measured for the phenotype get a made-up value and *falsely* match. |
| ✅ **OAKG** (observability-aware) | **unobserved / unknown** | Only patients that actually observed the phenotype are returned. |

Both panels are ranked by the **same relevance** (closeness to the condition's ideal), so the
observation-backed patients appear in the **same relative order** in both — the competitor panel just
injects extra ⚠️ false rows among them. The headline metric reports how many competitor hits are
**false positives**, and the *"who OAKG keeps vs what the competitor adds"* box shows, for the current
query, one real patient OAKG keeps next to one it drops.

## Llama 3.2 3B explainer + chatbot
- **📝 Explain these results** — generates a plain-language summary of the current query grounded in
  the retrieval facts.
- **Chat box** — ask follow-ups ("why did this competitor return more?", "which datasets caused the
  false positives?"). The model is fed a structured context (the selected competitor, dataset-level
  observability, false-positive counts by dataset, the leaderboard, example patients) and is
  instructed to ground only in those facts. **The conversation resets automatically when you change
  the query** (dataset, phenotype, condition, categorical filter, or competitor).
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
