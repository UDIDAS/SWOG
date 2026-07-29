#!/usr/bin/env python3
"""
OAKG phenotype-query demo (Streamlit) with a Llama 3.2 3B explainer/chatbot.

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
import statistics
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # make llm_backend importable


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
    return default if v is None else v


# ----------------------------------------------------------------------------- app
st.set_page_config(page_title="OAKG phenotype query", layout="wide")
records = load_records()

st.title("Patient retrieval by phenotype — OAKG vs coverage-blind")
st.caption(
    "Query the patient-level imaging KG by phenotype and a condition (e.g. *< 5 cm³*, *= 0*). The "
    "two panels differ only in how they treat **missing** values: coverage-blind imputes a constant "
    "(0 / mean / median → false matches); OAKG treats missing as **unobserved / unknown** and "
    "returns only reliable patients."
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
    st.markdown("**Numeric phenotype + condition**")
    num_key = st.selectbox("Phenotype", list(NUMERIC), format_func=lambda k: NUMERIC[k][0])
    nlabel, norgan, nkind, nfield = NUMERIC[num_key]
    obs_vals = [real_value(r, norgan, nfield) for r in cohort if is_observed(r, norgan, nkind)]
    obs_vals = [v for v in obs_vals if v is not None]
    vmax = float(max(obs_vals)) if obs_vals else 1.0

    OPERATORS = {"less than (<)": "<", "at most (≤)": "<=", "greater than (>)": ">",
                 "at least (≥)": ">=", "equals (=)": "==", "between (range)": "between"}
    SYM = {"<": "<", "<=": "≤", ">": ">", ">=": "≥", "==": "="}
    op = OPERATORS[st.selectbox("Condition", list(OPERATORS), index=0, key=f"op_{num_key}")]
    if op == "between":
        lo, hi = st.slider(f"{nlabel} range", 0.0, round(vmax, 1),
                           (0.0, round(vmax * 0.25, 1)), key=f"rng_{num_key}")
        def matches(v): return lo <= v <= hi
        target, cond_label = (lo + hi) / 2, f"in [{lo}, {hi}]"
    else:
        thr = st.number_input(f"{nlabel} — threshold", 0.0, round(vmax, 1),
                              0.0 if op == "==" else round(vmax * 0.25, 1), step=0.5,
                              key=f"thr_{num_key}_{op}")
        _ops = {"<": lambda v: v < thr, "<=": lambda v: v <= thr, ">": lambda v: v > thr,
                ">=": lambda v: v >= thr, "==": lambda v: abs(v - thr) < 1e-9}
        matches = _ops[op]
        target = {"<": 0.0, "<=": 0.0, ">": vmax, ">=": vmax, "==": thr}[op]
        cond_label = f"{SYM[op]} {thr}"
    st.caption(f"{len(obs_vals)} observed this phenotype (0–{vmax:.1f}); the other "
               f"{len(cohort)-len(obs_vals)} are UNOBSERVED for it.")

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

    st.header("Coverage-blind imputation")
    IMPUTE = {"zero (0)": 0.0,
              "cohort mean": round(statistics.mean(obs_vals), 2) if obs_vals else 0.0,
              "cohort median": round(statistics.median(obs_vals), 2) if obs_vals else 0.0}
    imp_label = st.selectbox("Fill a MISSING value with", list(IMPUTE), index=0)
    impute_value = IMPUTE[imp_label]
    st.caption("The non-OAKG baseline replaces every unobserved value with this constant. Any "
               "constant is a fabrication — it just changes *which* queries get false matches.")

    st.header("Display")
    n_rows = st.selectbox("Rows to show per panel", [5, 10, 15, 20, 30, 50, 75, 100, 150, 200],
                          index=1)


# ---- evaluate both engines ----
def relevance(v):
    """Closeness of a value to the condition's ideal point (1 = ideal, 0 = vmax away)."""
    if v is None:
        return 0.0
    return round(max(0.0, 1 - abs(v - target) / (vmax + 1e-9)), 3)


def eval_patient(rec):
    obs_n = is_observed(rec, norgan, nkind)
    rv = real_value(rec, norgan, nfield)
    nv = naive_value(rec, norgan, nfield, impute_value)
    num_oakg = obs_n and rv is not None and matches(rv)
    num_naive = matches(nv)
    imputed = num_naive and not obs_n                      # matched only via the imputed constant

    cat_oakg = cat_naive = True
    if cat_allowed is not None:
        clabel, corgan, ckind, cfield = CATEG[cat_key]
        cat_obs = is_observed(rec, corgan, ckind)
        cv = real_value(rec, corgan, cfield)
        ncv = naive_value(rec, corgan, cfield, "none")
        cat_oakg = cat_obs and cv in set(cat_allowed)
        cat_naive = ncv in set(cat_allowed)
        imputed = imputed or (cat_naive and not cat_obs)

    return {
        "patient": rec["case_id"], "dataset": rec["dataset"],
        "real": (round(rv, 2) if rv is not None else None),
        "seen": round(nv, 2), "observed": "observed" if obs_n else "UNOBSERVED",
        "relevance": relevance(rv if obs_n else nv),        # engine-appropriate rank key
        "oakg": num_oakg and cat_oakg, "naive": num_naive and cat_naive, "imputed": imputed,
    }


rows = [eval_patient(r) for r in cohort]
naive_rows = [r for r in rows if r["naive"]]
oakg_rows  = [r for r in rows if r["oakg"]]
false_rows = [r for r in rows if r["naive"] and r["imputed"] and not r["oakg"]]

# ---- headline metrics ----
st.subheader("Result")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Patients in cohort", len(cohort))
c2.metric("Observed for this phenotype", len(obs_vals))
c3.metric("Coverage-blind matches", len(naive_rows),
          delta=f"{len(false_rows)} false", delta_color="inverse")
c4.metric("OAKG matches (reliable)", len(oakg_rows))

if false_rows:
    st.error(f"⚠️ Coverage-blind returned **{len(false_rows)}** patients that never observed "
             f"*{nlabel}* — their value was filled with **{impute_value}** ({imp_label}), which "
             f"satisfies '{nlabel} {cond_label}'. OAKG excludes them.")
else:
    st.success(f"No imputation-driven false matches here: the fill value **{impute_value}** "
               f"({imp_label}) does **not** satisfy '{nlabel} {cond_label}', so unobserved patients "
               f"can't sneak in. Switch the condition (e.g. 'less than' / '= 0') or the fill value "
               f"to expose the problem.")

with st.expander("How to read the two panels (and why the ordering matches)"):
    st.markdown(
        "- **Both panels are ranked by *relevance*** — closeness of the value to the condition's "
        "ideal (1.0 = ideal match). So the genuinely-observed patients appear in the **same "
        "relative order** in both panels; that alignment is expected.\n"
        "- **❌ Coverage-blind** ranks *every* patient as if a missing value were the **imputed "
        "constant**. Rows tagged **⚠️ false (imputed)** matched only because an unobserved phenotype "
        "was filled in. These are the extra, unreliable rows interleaved among the real ones.\n"
        "- **✅ OAKG** ranks *only* patients that **actually observed** the phenotype (missing = "
        "unknown, never 0). Same real patients, none of the imputed noise — the reliable set.\n"
        "- If a real patient sits at a different absolute row number across panels, it is only "
        "because coverage-blind pushed imputed-zero patients in around it — not because its own "
        "score changed."
    )

# ---- side-by-side tables ----
left, right = st.columns(2)
with left:
    st.markdown(f"### ❌ Coverage-blind (missing → {impute_value})")
    st.caption(f"Treats an unmeasured phenotype as {impute_value} ({imp_label}). ⚠️-tagged rows are "
               f"false matches from imputation.")
    if naive_rows:
        df = pd.DataFrame(naive_rows).sort_values("relevance", ascending=False).head(n_rows)
        df.insert(0, "flag", ["⚠️ false (imputed)" if r["observed"] == "UNOBSERVED" else "✔ real"
                              for r in df.to_dict("records")])
        st.caption(f"showing top {min(n_rows, len(naive_rows))} of {len(naive_rows)} matches")
        st.dataframe(df.rename(columns={"seen": "value seen"})
                     [["flag", "patient", "dataset", "value seen", "relevance", "observed"]],
                     hide_index=True, width="stretch", height=430)
    else:
        st.info("No matches.")
with right:
    st.markdown("### ✅ OAKG (missing → unobserved)")
    st.caption("Returns only patients that actually observed the phenotype — no imputed-zero matches.")
    if oakg_rows:
        df = pd.DataFrame(oakg_rows).sort_values("relevance", ascending=False).head(n_rows)
        st.caption(f"showing top {min(n_rows, len(oakg_rows))} of {len(oakg_rows)} reliable matches")
        st.dataframe(df.rename(columns={"real": nlabel})
                     [["patient", "dataset", nlabel, "relevance"]],
                     hide_index=True, width="stretch", height=430)
    else:
        st.info("No patient reliably (observed) satisfies this query.")

# ---- concrete "kept vs dropped" on THIS query ----
with st.expander("On this query: who OAKG keeps vs drops", expanded=bool(false_rows)):
    kept_pool = [r for r in rows if r["oakg"]]                       # observed + in range
    kept = min(kept_pool, key=lambda r: r["real"]) if kept_pool else None  # smallest → nearest the imputed 0
    dropped = false_rows[0] if false_rows else None
    if kept and dropped:
        demo = pd.DataFrame([
            {"patient": kept["patient"], "dataset": kept["dataset"],
             f"{nlabel}": f"{kept['real']} — observed",
             "coverage-blind sees": f"{kept['seen']} (real)",
             "coverage-blind": "✓ returned", "OAKG": "✓ returned (reliable)"},
            {"patient": dropped["patient"], "dataset": dropped["dataset"],
             f"{nlabel}": "never measured (unobserved)",
             "coverage-blind sees": f"{dropped['seen']} (imputed)",
             "coverage-blind": "✓ returned (FALSE)", "OAKG": "✗ dropped"},
        ])
        st.table(demo)
        st.markdown(
            f"Both patients look like ~0 to **coverage-blind**, so it returns **both** — it cannot "
            f"tell a genuinely-measured small tumor (**{kept['patient']}**) from a patient whose "
            f"organ was never imaged (**{dropped['patient']}**, {dropped['dataset']}). "
            f"**OAKG** checks observability first: it keeps **{kept['patient']}** (measured) and "
            f"drops **{dropped['patient']}** (phenotype unknown, not 0). Apply that test to every "
            f"unobserved patient and all **{len(false_rows)}** false matches fall away — that is the "
            f"reliability gain, on this exact query."
        )
    elif kept and not dropped:
        st.success("Every returned patient actually observed this phenotype — coverage-blind and "
                   "OAKG agree here, so there is nothing for OAKG to drop.")
    else:
        st.info("No patient reliably satisfies this query, so there is no kept example to show.")

# ----------------------------------------------------------------------------- Llama 3.2 3B
st.divider()
st.subheader("🧠 More questions about these results")


@st.cache_resource(show_spinner="Loading Llama 3.2 3B (first use only)…")
def get_llm():
    from llm_backend import load_llama
    return load_llama()


def query_context():
    from collections import Counter
    top_oakg = sorted(oakg_rows, key=lambda r: r["relevance"], reverse=True)[:6]
    obs_ds = sorted({r["dataset"] for r in cohort if is_observed(r, norgan, nkind)})
    unobs_ds = sorted({r["dataset"] for r in cohort if not is_observed(r, norgan, nkind)})
    false_by_ds = Counter(r["dataset"] for r in false_rows)
    cat_txt = (f"{CATEG[cat_key][0]} in {cat_allowed}" if cat_allowed else "none")
    return (
        f"Phenotype queried: {nlabel}\n"
        f"Condition: {nlabel} {cond_label}   (categorical filter: {cat_txt})\n"
        f"Coverage-blind fills missing values with: {impute_value} ({imp_label}).\n"
        f"Cohort: {len(cohort)} patients ("
        + ", ".join(f"{d}={sum(1 for r in cohort if r['dataset']==d)}" for d in ds_sel) + ").\n"
        f"Datasets that OBSERVED this phenotype (eligible, reliable): {obs_ds or 'none'}.\n"
        f"Datasets that did NOT observe this phenotype (these are the source of false matches when "
        f"imputed to 0): {unobs_ds or 'none'}.\n"
        f"Patients that actually observed this phenotype: {len(obs_vals)}; the remaining "
        f"{len(cohort)-len(obs_vals)} are UNOBSERVED for it.\n"
        f"Coverage-blind retrieval (missing imputed to 0) returned {len(naive_rows)} patients, of "
        f"which {len(false_rows)} are FALSE matches. False matches by dataset: "
        f"{dict(false_by_ds) or 'none'} (all of these never observed the phenotype).\n"
        f"OAKG retrieval (missing treated as unobserved/unknown) returned {len(oakg_rows)} reliable "
        f"patients — every one comes from an OBSERVED dataset above.\n"
        f"Example reliable OAKG patients that DID observe it (id: value): "
        + (", ".join(f"{r['patient']}: {r['real']}" for r in top_oakg) or "none")
    )


SYSTEM = (
    "You are a data assistant explaining patient-retrieval results from an imaging knowledge graph "
    "to a clinical research team. Be concise, clear, and grounded ONLY in the CONTEXT provided; do "
    "not invent patients, datasets, or numbers. RULES: (1) False matches come ONLY from the datasets "
    "listed as 'did NOT observe this phenotype' — never call an OAKG/observed patient a false match. "
    "(2) The 'reliable OAKG patients' listed DID observe the phenotype; they are correct, not false. "
    "Key idea: OAKG (observability-aware) treats a missing phenotype as UNOBSERVED/unknown, whereas "
    "the coverage-blind method imputes it to 0 — creating false matches and making truly-unmeasured "
    "patients look identical to genuinely-zero ones. Prefer plain clinical language."
)

st.session_state.setdefault("chat", [])

b1, b2 = st.columns([1, 1])
if b1.button("📝 Explain these results", width="stretch"):
    with st.spinner("Llama 3.2 3B is thinking…"):
        from llm_backend import generate
        tok, model, _ = get_llm()
        msg = [{"role": "system", "content": SYSTEM},
               {"role": "user", "content": "CONTEXT:\n" + query_context()
                + "\n\nWrite a short paragraph explaining what this query returned and why the two "
                  "methods disagree, for a clinician reading the dashboard."}]
        st.session_state.chat.append(
            ("assistant", generate(tok, model, msg, max_new_tokens=320, temperature=0.2)))
if b2.button("🗑 Clear chat", width="stretch"):
    st.session_state.chat = []

for role, text in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(text)

prompt = st.chat_input("Ask about the retrieval (e.g. 'why did coverage-blind return more?')")
if prompt:
    st.session_state.chat.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"), st.spinner("Llama 3.2 3B is thinking…"):
        from llm_backend import generate
        tok, model, repo = get_llm()
        history = [{"role": r, "content": t} for r, t in st.session_state.chat if r in ("user", "assistant")]
        msgs = [{"role": "system", "content": SYSTEM + "\n\nCONTEXT (current query):\n" + query_context()}] + history
        reply = generate(tok, model, msgs, max_new_tokens=350, temperature=0.2)
        st.markdown(reply)
        st.session_state.chat.append(("assistant", reply))
        st.caption(f"powered by {repo}")

st.caption(f"KG source: {os.path.relpath(CORPUS, ROOT)} · {len(records)} patient instances · "
           "observability derived from `observed_organs` + dataset tumor-annotation coverage.")
