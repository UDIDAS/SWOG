# OAKG phenotype-query demo

A minimal Streamlit app that retrieves patients from the imaging KG by **phenotype + desirable
range**, run two ways side by side to show why **observability-awareness (OAKG)** is more reliable
than a coverage-blind retriever that imputes missing values to 0.

## Run
```bash
conda activate llmft            # env with streamlit + pandas
streamlit run app/oakg_query_app.py
```
Opens at http://localhost:8501. No data prep — it reads `kg/data/corpus_perpatient.json`
(512 patient-level KG instances: 281 Pancreas, 131 LiTS, 100 FLARE).

## What it shows
Pick a numeric phenotype (e.g. *Pancreatic tumor volume*) and a range, optionally add a categorical
filter. Two panels return patients:

| Panel | Missing value = | Effect |
|---|---|---|
| ❌ **Coverage-blind** (non-OAKG) | **0** (imputed) | Patients never measured for the phenotype get 0 and *falsely* match "small/low" ranges. |
| ✅ **OAKG** (observability-aware) | **unobserved / unknown** | Only patients that actually observed the phenotype are returned. |

The headline metric reports how many coverage-blind hits are **false matches** (matched only because
a missing value was imputed to 0), and a callout contrasts a **true observed zero** (organ imaged,
tumor genuinely absent) with a **false zero** (organ never imaged) — the two are indistinguishable to
the coverage-blind retriever but correctly separated by OAKG.

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
