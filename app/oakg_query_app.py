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
import streamlit.components.v1 as components

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # make sibling modules importable
import paper_retrieval as pr   # noqa: E402  (needs the path insert above)


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


@st.cache_data
def load_mappings():
    return json.load(open(os.path.join(ROOT, "kg", "ontology_mappings.json")))["mappings"]


@st.cache_resource
def load_corpus():
    return pr.build_corpus(load_records())


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
rec_by_id = {r["case_id"]: r for r in records}

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

    st.header("Competitor (vs OAKG)")
    mean_v = round(statistics.mean(obs_vals), 2) if obs_vals else 0.0
    median_v = round(statistics.median(obs_vals), 2) if obs_vals else 0.0
    COMPETITORS = {
        "Zero imputation (missing → 0)":
            ("impute", 0.0, "unobserved value filled with 0"),
        "Mean imputation (missing → cohort mean)":
            ("impute", mean_v, f"unobserved value filled with the cohort mean ({mean_v})"),
        "Median imputation (missing → cohort median)":
            ("impute", median_v, f"unobserved value filled with the cohort median ({median_v})"),
        "Cross-organ collision (untyped phenotype)":
            ("cross_organ", None, "any organ's value answers the query — right number, wrong organ"),
    }
    comp_label = st.selectbox("Compare OAKG against", list(COMPETITORS), index=0)
    comp_code, comp_const, comp_src = COMPETITORS[comp_label]
    comp_short = comp_label.split(" (")[0]
    st.caption("OAKG is the guardrail (observability-aware). The competitor is a common baseline that "
               "fabricates a value for unobserved phenotypes — each fabrication raises false positives "
               "on a different set of queries. The leaderboard below scores all of them at once.")

    st.header("Display")
    n_rows = st.selectbox("Rows to show per panel", [5, 10, 15, 20, 30, 50, 75, 100, 150, 200],
                          index=1)


# ---- retrieval engines: OAKG guardrail vs a selectable competitor ----
def relevance(v):
    """Closeness of a value to the condition's ideal point (1 = ideal, 0 = vmax away)."""
    if v is None:
        return 0.0
    return round(max(0.0, 1 - abs(v - target) / (vmax + 1e-9)), 3)


def cat_ok_oakg(rec):
    if cat_allowed is None:
        return True
    _, corgan, ckind, cfield = CATEG[cat_key]
    return is_observed(rec, corgan, ckind) and real_value(rec, corgan, cfield) in set(cat_allowed)


def cat_ok_comp(rec):
    if cat_allowed is None:
        return True
    _, corgan, ckind, cfield = CATEG[cat_key]
    return naive_value(rec, corgan, cfield, "none") in set(cat_allowed)


def oakg_match(rec):
    rv = real_value(rec, norgan, nfield)
    return (is_observed(rec, norgan, nkind) and rv is not None
            and matches(rv) and cat_ok_oakg(rec))


def comp_seen(rec, code, const):
    """The value a competitor 'uses' for the queried phenotype (display + matching)."""
    if code == "cross_organ":                       # borrow ANY organ's value for this field
        vals = [od.get(nfield) for od in rec["organs"].values() if od.get(nfield) is not None]
        return next((v for v in vals if matches(v)), vals[0] if vals else 0.0)
    return naive_value(rec, norgan, nfield, const)  # else fill missing with the imputed constant


def comp_match(rec, code, const):
    if code == "cross_organ":
        vals = [od.get(nfield) for od in rec["organs"].values() if od.get(nfield) is not None]
        num = any(matches(v) for v in vals)
    else:
        num = matches(naive_value(rec, norgan, nfield, const))
    return num and cat_ok_comp(rec)


oakg_ids = {r["case_id"] for r in cohort if oakg_match(r)}   # observation-backed reference set


def row_of(rec):
    obs_n = is_observed(rec, norgan, nkind)
    rv = real_value(rec, norgan, nfield)
    seen = comp_seen(rec, comp_code, comp_const)
    oak = rec["case_id"] in oakg_ids
    comp = comp_match(rec, comp_code, comp_const)
    seen_num = seen if isinstance(seen, (int, float)) else None
    return {
        "patient": rec["case_id"], "dataset": rec["dataset"],
        "real": (round(rv, 2) if rv is not None else None),
        "seen": (round(seen, 2) if seen_num is not None else seen),
        "observed": "observed" if obs_n else "UNOBSERVED",
        "relevance": relevance(rv if oak else seen_num),
        "oakg": oak, "comp": comp, "false": comp and not oak,
    }


rows = [row_of(r) for r in cohort]
comp_rows = [r for r in rows if r["comp"]]
oakg_rows = [r for r in rows if r["oakg"]]
false_rows = [r for r in rows if r["false"]]

# ---- retrieval-quality leaderboard: OAKG vs ALL competitors ----
leader = [{"method": "🟢 OAKG (guardrail)", "returned": len(oakg_ids), "false positives": 0,
           "precision": 1.0, "false-positive source": "— reference (observation-backed)"}]
for lab, (code, const, src) in COMPETITORS.items():
    ret = {r["case_id"] for r in cohort if comp_match(r, code, const)}
    prec = round(len(ret & oakg_ids) / len(ret), 3) if ret else 1.0
    leader.append({"method": ("▶ " if lab == comp_label else "") + lab, "returned": len(ret),
                   "false positives": len(ret - oakg_ids), "precision": prec,
                   "false-positive source": src})

# ---- headline metrics ----
st.subheader("Result")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Patients in cohort", len(cohort))
c2.metric("Observed for this phenotype", len(obs_vals))
c3.metric(f"{comp_short} matches", len(comp_rows),
          delta=f"{len(false_rows)} false", delta_color="inverse")
c4.metric("OAKG matches (reliable)", len(oakg_rows))

if false_rows:
    st.error(f"⚠️ **{comp_short}** returned **{len(false_rows)}** false positives for "
             f"'{nlabel} {cond_label}' — not observation-backed ({comp_src}). OAKG excludes them.")
else:
    st.success(f"**{comp_short}** produced no false positives for '{nlabel} {cond_label}' in this "
               f"cohort — its fabricated value doesn't satisfy the condition here. Try 'less than' / "
               f"'= 0', or another competitor, to expose it.")

with st.expander("Retrieval-quality leaderboard — OAKG vs every competitor", expanded=True):
    st.dataframe(pd.DataFrame(leader).sort_values("precision", ascending=False),
                 hide_index=True, width="stretch")
    st.caption("**Precision** = fraction of a method's returned patients that are observation-backed "
               "(agree with OAKG). OAKG is the reference (1.00 by construction); every competitor "
               "drops below 1.00 exactly when its fabricated value satisfies the condition. ▶ marks "
               "the competitor shown on the left below.")

with st.expander("How to read the two panels (and why the ordering matches)"):
    st.markdown(
        "- **Both panels are ranked by *relevance*** — closeness of the value to the condition's "
        "ideal (1.0 = ideal match). So the observation-backed patients appear in the **same relative "
        "order** in both panels; that alignment is expected.\n"
        f"- **❌ {comp_short}** ranks *every* patient using a **fabricated** value where the phenotype "
        f"is missing ({comp_src}). Rows tagged **⚠️ false** are the ones not backed by an actual "
        "observation — extra, unreliable rows interleaved among the real ones.\n"
        "- **✅ OAKG** ranks *only* patients that **actually observed** the phenotype (missing = "
        "unknown, never fabricated). Same real patients, none of the noise — the reliable set.\n"
        "- If a real patient sits at a different absolute row number across panels, it is only "
        "because the competitor pushed fabricated matches in around it — not because its own score "
        "changed."
    )

# ---- side-by-side tables ----
left, right = st.columns(2)
with left:
    st.markdown(f"### ❌ {comp_short}")
    st.caption(f"{comp_src}. ⚠️-tagged rows are false positives (not observation-backed).")
    if comp_rows:
        df = pd.DataFrame(comp_rows).sort_values("relevance", ascending=False).head(n_rows)
        df.insert(0, "flag", ["⚠️ false" if r["false"] else "✔ real" for r in df.to_dict("records")])
        st.caption(f"showing top {min(n_rows, len(comp_rows))} of {len(comp_rows)} matches")
        st.dataframe(df.rename(columns={"seen": "value used"})
                     [["flag", "patient", "dataset", "value used", "relevance", "observed"]],
                     hide_index=True, width="stretch", height=430)
    else:
        st.info("No matches.")
with right:
    st.markdown("### ✅ OAKG (missing → unobserved)")
    st.caption("Returns only patients that actually observed the phenotype — no fabricated matches.")
    if oakg_rows:
        df = pd.DataFrame(oakg_rows).sort_values("relevance", ascending=False).head(n_rows)
        st.caption(f"showing top {min(n_rows, len(oakg_rows))} of {len(oakg_rows)} reliable matches")
        st.dataframe(df.rename(columns={"real": nlabel})
                     [["patient", "dataset", nlabel, "relevance"]],
                     hide_index=True, width="stretch", height=430)
    else:
        st.info("No patient reliably (observed) satisfies this query.")

# ---- concrete "kept vs dropped" on THIS query ----
with st.expander(f"On this query: who OAKG keeps vs what {comp_short} adds", expanded=bool(false_rows)):
    kept_pool = [r for r in rows if r["oakg"]]
    kept = min(kept_pool, key=lambda r: (r["real"] if r["real"] is not None else 0)) if kept_pool else None
    dropped = false_rows[0] if false_rows else None
    if kept and dropped:
        demo = pd.DataFrame([
            {"patient": kept["patient"], "dataset": kept["dataset"],
             f"{nlabel}": f"{kept['real']} — observed",
             f"{comp_short} used": f"{kept['seen']}",
             comp_short: "✓ returned", "OAKG": "✓ returned (reliable)"},
            {"patient": dropped["patient"], "dataset": dropped["dataset"],
             f"{nlabel}": "not observation-backed",
             f"{comp_short} used": f"{dropped['seen']}",
             comp_short: "✓ returned (FALSE)", "OAKG": "✗ dropped"},
        ])
        st.table(demo)
        st.markdown(
            f"**{comp_short}** returns both — it can't tell an observation-backed match "
            f"(**{kept['patient']}**) from one that isn't (**{dropped['patient']}**, "
            f"{dropped['dataset']}; {comp_src}). **OAKG** checks observability first: it keeps "
            f"**{kept['patient']}** and drops **{dropped['patient']}**. Across the cohort that removes "
            f"all **{len(false_rows)}** false positives for this query."
        )
    elif kept and not dropped:
        st.success(f"Every patient {comp_short} returned is observation-backed — it and OAKG agree "
                   "on this query, so there is nothing to drop.")
    else:
        st.info("No patient reliably satisfies this query, so there is no kept example to show.")

# ------------------------------------------------- similarity retrieval (OAKG paper baselines)
st.divider()
st.subheader("🔬 Similarity retrieval — OAKG vs the paper's baselines (Section 5)")
st.caption("The exact competitors from the OAKG paper. Task: rank a **query patient** against all "
           "others over a masked phenotype matrix. OAKG restricts to jointly-observed features and "
           "weights by shared-evidence **γ** (Jaccard of observed organs); the baselines don't — so "
           "a patient sharing only one feature can look like a perfect match (γ is low).")
corpus, Xm, Mm = load_corpus()
s1, s2, s3 = st.columns([2, 2, 1])
q_labels = [f"{r['case_id']}  ·  {r['dataset']}" for r in records]
qcase = s1.selectbox("Query patient", q_labels, key="sim_q").split("  ·  ")[0]
base = s2.selectbox("Paper baseline vs OAKG", list(pr.BASELINE_FUNCTIONS), index=3)  # Masked cosine
topk = s3.selectbox("top-k", [5, 10, 15, 20], index=1)
blind = ("⚠️ coverage-blind (fabricates missing values)" if pr.COVERAGE_BLIND[base]
         else "coverage-aware (uses only jointly-observed features)")

oakg_top = pr.rank(qcase, "OAKG", corpus, Xm, Mm, records, topk)
base_top = pr.rank(qcase, base, corpus, Xm, Mm, records, topk)
low_oakg = sum(t["low evidence (γ<0.25)"] for t in oakg_top)
low_base = sum(t["low evidence (γ<0.25)"] for t in base_top)

k1, k2, k3 = st.columns(3)
k1.metric("Query observed organs", ", ".join(sorted(corpus.observation_sets[qcase])) or "—")
k2.metric(f"OAKG low-evidence in top-{topk}", low_oakg)
k3.metric(f"{base} low-evidence in top-{topk}", low_base,
          delta=f"{blind}", delta_color="off")


def sim_table(rows):
    df = pd.DataFrame(rows)
    df.insert(0, "flag", ["⚠️ weak match (γ<0.25)" if r["low evidence (γ<0.25)"] else "✔"
                          for r in rows])
    return df[["flag", "patient", "dataset", "score", "shared organs", "γ"]]


colL, colR = st.columns(2)
with colL:
    st.markdown(f"### ❌ {base}")
    st.caption(blind)
    st.dataframe(sim_table(base_top), hide_index=True, width="stretch", height=380)
with colR:
    st.markdown("### ✅ OAKG (support-restricted + γ)")
    st.caption("Only jointly-observed, anatomically-supported features; γ-weighted so weak-overlap "
               "neighbors sink.")
    st.dataframe(sim_table(oakg_top), hide_index=True, width="stretch", height=380)

if low_base > low_oakg:
    st.error(f"**{base}** put **{low_base}** weak-overlap neighbours (γ<0.25) in its top-{topk} — "
             f"patients that share almost no observed evidence with {qcase}. OAKG has {low_oakg}: its "
             "shared-evidence γ weight pushes those down.")
else:
    st.success(f"On this query, {base} and OAKG agree on evidence quality (weak-overlap neighbours: "
               f"{base} {low_base}, OAKG {low_oakg}).")

with st.expander("All paper baselines — weak-overlap neighbours in the top-k (lower = better)"):
    board = [{"method": "🟢 OAKG", "weak-overlap neighbours (γ<0.25)": low_oakg,
              "coverage-blind": "—"}]
    for m in pr.BASELINE_FUNCTIONS:
        rk = pr.rank(qcase, m, corpus, Xm, Mm, records, topk)
        board.append({"method": m,
                      "weak-overlap neighbours (γ<0.25)": sum(r["low evidence (γ<0.25)"] for r in rk),
                      "coverage-blind": "yes" if pr.COVERAGE_BLIND[m] else "no (masked)"})
    st.dataframe(pd.DataFrame(board), hide_index=True, width="stretch")
    st.caption("Baselines vendored verbatim from the OAKG paper (`oakg/baselines.py`). Zero/Mean/"
               "Missingness are coverage-blind; Masked cosine & Gower are coverage-aware but "
               "un-weighted, so tiny overlaps still score high — OAKG's γ is what fixes that.")

# ----------------------------------------------------------------------------- KG visualization
st.divider()
st.subheader("🕸 KG visualization — the OAKG-retrieved (reliable) patients")
if not oakg_rows:
    st.info("Run a query that returns OAKG patients to populate the visualizations.")
else:
    import kg_viz
    mappings = load_mappings()
    ranked = sorted(oakg_rows, key=lambda r: r["relevance"], reverse=True)
    retrieved = [rec_by_id[r["patient"]] for r in ranked]
    st.caption(f"Visualizing the **{len(retrieved)}** observation-backed patients OAKG returned — "
               "the reliable set only, no fabricated matches.")

    st.markdown("**Retrieved cohort at a glance**")
    v1, v2 = st.columns(2)
    v1.plotly_chart(kg_viz.cohort_bar_figure(retrieved), width="stretch")
    v2.plotly_chart(kg_viz.cohort_strip_figure(retrieved, nlabel, norgan, nfield), width="stretch")

    st.markdown("**Per-patient knowledge graph** — drag nodes, hover for details, scroll to zoom")
    opt_labels = [f"{r['patient']}  ·  {rec_by_id[r['patient']]['dataset']}" for r in ranked]
    choice = st.selectbox("Patient (ranked by relevance)", opt_labels,
                          key=f"vizpat_{hash(tuple(opt_labels))}")
    rec = rec_by_id[choice.split("  ·  ")[0]]
    st.markdown(" &nbsp; ".join(f"<span style='color:{c};font-size:18px'>●</span> {t}"
                                for t, c in kg_viz.TYPE_COLOR.items()), unsafe_allow_html=True)
    components.html(kg_viz.patient_graph_html(rec, mappings, height=560), height=584)
    bcol, _ = st.columns([1, 2])
    bcol.plotly_chart(kg_viz.patient_bars_figure(rec, norgan), width="stretch")
    with st.expander("Full record (raw KG properties)"):
        st.json(rec)

# ----------------------------------------------------------------------------- Llama 3.2 3B
st.divider()
st.subheader("🧠 Ask Llama 3.2 3B about these results")
st.caption("Grounded in the current query, the selected competitor, and the leaderboard. "
           "The conversation **resets automatically when you change the query.**")


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
    board = " | ".join(f"{d['method']}: returned={d['returned']}, false={d['false positives']}, "
                       f"precision={d['precision']}" for d in leader)
    return (
        f"Condition: {nlabel} {cond_label}   (categorical filter: {cat_txt})\n"
        f"Cohort: {len(cohort)} patients ("
        + ", ".join(f"{d}={sum(1 for r in cohort if r['dataset']==d)}" for d in ds_sel) + ").\n"
        f"Datasets that OBSERVED this phenotype (reliable): {obs_ds or 'none'}.\n"
        f"Datasets that did NOT observe it (source of false positives): {unobs_ds or 'none'}.\n"
        f"OAKG (observability guardrail) returned {len(oakg_ids)} reliable patients — all "
        f"observation-backed.\n"
        f"Selected competitor: {comp_label} — {comp_src}. It returned {len(comp_rows)} patients, "
        f"{len(false_rows)} of them FALSE positives (by dataset: {dict(false_by_ds) or 'none'}).\n"
        f"Retrieval-quality leaderboard (higher precision = fewer false positives): {board}.\n"
        f"Example reliable OAKG patients (id: value): "
        + (", ".join(f"{r['patient']}: {r['real']}" for r in top_oakg) or "none")
    )


SYSTEM = (
    "You are a data assistant explaining patient-retrieval results from an imaging knowledge graph "
    "to a clinical research team. Be concise, clear, and grounded ONLY in the CONTEXT provided; do "
    "not invent patients, datasets, or numbers. RULES: (1) False positives come ONLY from the "
    "competitor method and the datasets listed as 'did NOT observe this phenotype' — never call an "
    "OAKG/observation-backed patient a false positive. (2) 'Precision' in the leaderboard = fraction "
    "of a method's returns that are observation-backed; OAKG is the reference at 1.00. "
    "Key idea: OAKG (observability-aware) treats a missing phenotype as UNOBSERVED/unknown, whereas "
    "competitors fabricate a value (impute a constant, or borrow another organ's value) — which "
    "creates false positives. Prefer plain clinical language."
)

# reset the chat whenever the QUERY changes (new query -> fresh conversation)
query_sig = (tuple(sorted(ds_sel)), num_key, cond_label, cat_key,
             tuple(cat_allowed or ()), comp_label)
if st.session_state.get("last_sig") != query_sig:
    st.session_state.chat = []
    st.session_state.last_sig = query_sig
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
