#!/usr/bin/env python3
"""
OAKG phenotype-query demo (Streamlit).

Patient retrieval over the patient-level imaging KG by phenotype + desirable range, run TWO ways
side by side so you can see why observability-awareness matters:

  • Coverage-blind (non-OAKG)  — a missing phenotype is imputed to 0. Patients who were never
    measured for the queried phenotype get value 0 and *falsely* match "small / low" ranges. A
    genuinely-tumor-free patient (observed 0) and a never-imaged patient (unobserved) look
    identical → the retriever is "confused as similar" by shared missing values.

  • OAKG (observability-aware) — a missing phenotype is UNKNOWN, not 0. A patient can only be
    returned for a phenotype it actually observed. No false matches from imputed zeros.

The observability is REAL, straight from the KG instances (kg/data/corpus_perpatient.json):
  - Pancreas patients (281) imaged only the pancreas  → liver/spleen/kidney UNOBSERVED.
  - LiTS patients (131) imaged only the liver; 24 have an OBSERVED-zero liver tumor (imaged, none).
  - FLARE patients (100) imaged 5 organ volumes but tumors were never annotated (organs-only
    dataset) → their tumor_volume 0.0 is a FALSE zero (UNOBSERVED), not a real absence.

Run:  streamlit run app/oakg_query_app.py
"""
import json
import os

import pandas as pd
import streamlit as st


# ----------------------------------------------------------------------------- data
def find_root():
    d = os.path.abspath(os.path.dirname(__file__))
    for _ in range(6):
        if os.path.exists(os.path.join(d, "kg", "data", "corpus_perpatient.json")):
            return d
        d = os.path.dirname(d)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


ROOT = find_root()
CORPUS = os.path.join(ROOT, "kg", "data", "corpus_perpatient.json")

# phenotype registry:  key -> (label, organ, kind, field)
NUMERIC = {
    "panc_tumor_vol":  ("Pancreatic tumor volume (cm³)", "pancreas",     "tumor", "tumor_volume_cm3"),
    "liver_tumor_vol": ("Liver tumor volume (cm³)",      "liver",        "tumor", "tumor_volume_cm3"),
    "panc_vol":        ("Pancreas volume (cm³)",         "pancreas",     "organ", "organ_volume_cm3"),
    "liver_vol":       ("Liver volume (cm³)",            "liver",        "organ", "organ_volume_cm3"),
    "spleen_vol":      ("Spleen volume (cm³)",           "spleen",       "organ", "organ_volume_cm3"),
    "rkid_vol":        ("Right kidney volume (cm³)",     "right_kidney", "organ", "organ_volume_cm3"),
    "lkid_vol":        ("Left kidney volume (cm³)",      "left_kidney",  "organ", "organ_volume_cm3"),
}
CATEG = {
    "panc_burden": ("Pancreatic tumor burden",        "pancreas", "tumor", "burden_cat"),
    "panc_mult":   ("Pancreatic tumor multiplicity",  "pancreas", "tumor", "multiplicity"),
    "panc_loc":    ("Pancreatic tumor location",      "pancreas", "tumor", "anatomic_location"),
    "liver_burden":("Liver tumor burden",             "liver",    "tumor", "burden_cat"),
    "liver_mult":  ("Liver tumor multiplicity",       "liver",    "tumor", "multiplicity"),
}
# a dataset annotates an organ's TUMOR only if it is that organ's specialist set
TUMOR_ANNOTATED = {("pancreas", "pancreas"), ("liver", "lits")}


@st.cache_data
def load_records():
    return json.load(open(CORPUS))["records"]


def is_observed(rec, organ, kind):
    """Was this phenotype actually measured for this patient?"""
    if organ not in rec["observed_organs"]:
        return False                       # organ never imaged
    if kind == "organ":
        return True                        # organ volume is observed
    return (organ, rec["dataset"]) in TUMOR_ANNOTATED   # tumor observed only where annotated


def real_value(rec, organ, field):
    od = rec["organs"].get(organ)
    return None if od is None else od.get(field)


def naive_value(rec, organ, field, default):
    """What a coverage-blind reader sees: the stored value, else the imputed default (0 / 'none')."""
    od = rec["organs"].get(organ)
    if od is None:
        return default                     # organ absent from record -> imputed
    v = od.get(field)
    return default if v in (None,) else v


# ----------------------------------------------------------------------------- app
st.set_page_config(page_title="OAKG phenotype query", layout="wide")
records = load_records()

st.title("Patient retrieval by phenotype — OAKG vs coverage-blind")
st.caption(
    "Query the patient-level imaging KG by phenotype and a desirable range. The two panels differ "
    "only in how they treat **missing** values: coverage-blind imputes 0 (false match); OAKG treats "
    "missing as **unobserved / unknown** and returns only reliable patients."
)

# ---- sidebar: cohort + query builder ----
with st.sidebar:
    st.header("Cohort")
    ds_all = sorted({r["dataset"] for r in records})
    ds_sel = st.multiselect("Datasets in the KG", ds_all, default=ds_all)
    cohort = [r for r in records if r["dataset"] in ds_sel]
    st.caption(f"{len(cohort)} patients loaded "
               + ", ".join(f"{d}={sum(1 for r in cohort if r['dataset']==d)}" for d in ds_sel))

    st.header("Query")
    st.markdown("**Numeric phenotype + desirable range**")
    num_key = st.selectbox("Phenotype", list(NUMERIC),
                           format_func=lambda k: NUMERIC[k][0])
    nlabel, norgan, nkind, nfield = NUMERIC[num_key]
    obs_vals = [real_value(r, norgan, nfield) for r in cohort if is_observed(r, norgan, nkind)]
    obs_vals = [v for v in obs_vals if v is not None]
    vmax = float(max(obs_vals)) if obs_vals else 1.0
    rng = st.slider(f"{nlabel} — keep patients in range", 0.0, round(vmax, 1),
                    (0.0, round(vmax * 0.25, 1)), key=f"rng_{num_key}")
    st.caption(f"{len(obs_vals)} patients actually observed this phenotype "
               f"(range 0–{vmax:.1f}). The other {len(cohort)-len(obs_vals)} are UNOBSERVED for it.")

    st.markdown("**Optional categorical filter**")
    cat_key = st.selectbox("Add a categorical constraint", ["(none)"] + list(CATEG),
                           format_func=lambda k: k if k == "(none)" else CATEG[k][0])
    cat_allowed = None
    if cat_key != "(none)":
        clabel, corgan, ckind, cfield = CATEG[cat_key]
        opts = sorted({real_value(r, corgan, cfield) for r in cohort
                       if is_observed(r, corgan, ckind) and real_value(r, corgan, cfield)
                       not in (None, "unknown", "none", "na")})
        cat_allowed = st.multiselect(f"{clabel} in", opts, default=opts[:1] if opts else [])


# ---- evaluate both engines ----
def eval_patient(rec):
    """Return (oakg_match, naive_match, uses_imputed, row) for the current query."""
    obs_n = is_observed(rec, norgan, nkind)
    rv = real_value(rec, norgan, nfield)
    nv = naive_value(rec, norgan, nfield, 0.0)
    num_oakg = obs_n and rv is not None and rng[0] <= rv <= rng[1]
    num_naive = rng[0] <= nv <= rng[1]
    uses_imputed = num_naive and not obs_n            # matched only because missing->0

    cat_oakg = cat_naive = True
    cat_obs = True
    if cat_allowed is not None:
        clabel, corgan, ckind, cfield = CATEG[cat_key]
        cat_obs = is_observed(rec, corgan, ckind)
        cv = real_value(rec, corgan, cfield)
        ncv = naive_value(rec, corgan, cfield, "none")
        cat_oakg = cat_obs and cv in set(cat_allowed)
        cat_naive = ncv in set(cat_allowed)
        uses_imputed = uses_imputed or (cat_naive and not cat_obs)

    oakg = num_oakg and cat_oakg
    naive = num_naive and cat_naive
    # relevance: closeness of the observed numeric value to the range centre
    centre = (rng[0] + rng[1]) / 2
    relevance = round(1 - abs((rv if rv is not None else centre) - centre) / (vmax + 1e-9), 3)
    row = {
        "patient": rec["case_id"], "dataset": rec["dataset"],
        nlabel: (round(rv, 2) if rv is not None else None),
        "observed?": "observed" if obs_n else "UNOBSERVED",
        "coverage-blind sees": round(nv, 2),
        "relevance": relevance,
    }
    return oakg, naive, uses_imputed, row


results = [eval_patient(r) for r in cohort]
oakg_rows   = [r for o, n, imp, r in results if o]
naive_rows  = [r for o, n, imp, r in results if n]
false_rows  = [r for o, n, imp, r in results if n and imp and not o]   # false matches (imputed)

# ---- headline metrics ----
st.subheader("Result")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Patients in cohort", len(cohort))
c2.metric("Observed for this phenotype", len(obs_vals))
c3.metric("Coverage-blind matches", len(naive_rows),
          delta=f"{len(false_rows)} false", delta_color="inverse")
c4.metric("OAKG matches (reliable)", len(oakg_rows))

if false_rows:
    st.error(
        f"⚠️ Coverage-blind returned **{len(false_rows)}** patients that never observed "
        f"*{nlabel}* — their value was imputed to 0 and fell in range. OAKG excludes them."
    )
else:
    st.success("No imputation-driven false matches for this query in the current cohort.")

# ---- side-by-side tables ----
left, right = st.columns(2)
with left:
    st.markdown("### ❌ Coverage-blind (missing → 0)")
    if naive_rows:
        df = pd.DataFrame(naive_rows).sort_values("coverage-blind sees")
        df.insert(0, "flag", ["⚠️ false (imputed)" if r["observed?"] == "UNOBSERVED"
                              else "✔ real" for r in df.to_dict("records")])
        st.dataframe(df[["flag", "patient", "dataset", nlabel, "coverage-blind sees", "observed?"]],
                     hide_index=True, width="stretch", height=430)
    else:
        st.info("No matches.")
with right:
    st.markdown("### ✅ OAKG (missing → unobserved)")
    if oakg_rows:
        df = pd.DataFrame(oakg_rows).sort_values("relevance", ascending=False)
        st.dataframe(df[["patient", "dataset", nlabel, "relevance"]],
                     hide_index=True, width="stretch", height=430)
    else:
        st.info("No patient reliably (observed) satisfies this query.")

# ---- true-zero vs false-zero teaching callout ----
with st.expander("Why they differ — a TRUE zero vs a FALSE zero (the core point)", expanded=bool(false_rows)):
    obs_here = [r for r in cohort if is_observed(r, norgan, nkind)
                and real_value(r, norgan, nfield) is not None]
    true_zero = next((r for r in obs_here if real_value(r, norgan, nfield) == 0), None)
    if true_zero is None and obs_here:      # no exact 0 -> use the smallest real measurement
        true_zero = min(obs_here, key=lambda r: real_value(r, norgan, nfield))
    false_zero = next((r for r in cohort if not is_observed(r, norgan, nkind)), None)
    cc = st.columns(2)
    with cc[0]:
        st.markdown(f"**Observed** — *{nlabel}* was actually measured.")
        if true_zero:
            tv = real_value(true_zero, norgan, nfield)
            st.json({"patient": true_zero["case_id"], "dataset": true_zero["dataset"],
                     "observed_organs": true_zero["observed_organs"], nfield: tv,
                     "status": "observed → genuine 0" if tv == 0 else f"observed → real {tv} cm³"})
        else:
            st.caption("No observed example for this phenotype in the cohort.")
    with cc[1]:
        st.markdown(f"**Unobserved (false zero)** — *{nlabel}* was never measured.")
        if false_zero:
            st.json({"patient": false_zero["case_id"], "dataset": false_zero["dataset"],
                     "observed_organs": false_zero["observed_organs"],
                     "coverage-blind imputes": 0.0, "status": "UNOBSERVED → not a real 0"})
        else:
            st.caption("Every patient observed this phenotype in the current cohort.")
    st.markdown(
        "Coverage-blind assigns **both** the value 0 and calls them *similar / matching*. "
        "OAKG keeps the right one and drops the unobserved one — that is the reliability gain."
    )

st.caption(f"KG source: {os.path.relpath(CORPUS, ROOT)} · {len(records)} patient instances · "
           "observability derived from `observed_organs` + dataset tumor-annotation coverage.")
