#!/usr/bin/env python3
"""
OAKG similarity-retrieval demo (Streamlit) with a Llama 3.2 3B explainer/chatbot.

Everything is driven from the sidebar:
  1. Cohort (datasets).
  2. Anchor patient — found with a phenotype range query. OAKG returns only observation-backed
     matches; you pick one as the ANCHOR.
  3. Similarity comparison — which OAKG-paper baseline to pit against OAKG.

The anchor drives the main page: similarity retrieval (OAKG vs the paper's baselines), an interactive
KG view of the anchor, and a Llama 3.2 3B explainer. The paper baselines + OAKG scorer are vendored
verbatim in paper_retrieval.py; observability is real (from `observed_organs` + dataset tumor
annotation): Pancreas patients imaged only the pancreas, LiTS only the liver, FLARE 5 organ volumes
but no tumors.

Run:  streamlit run app/oakg_query_app.py
"""
import json
import os
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
    if organ not in rec["observed_organs"]:
        return False
    if kind == "organ":
        return True
    return (organ, rec["dataset"]) in TUMOR_ANNOTATED


def real_value(rec, organ, field):
    od = rec["organs"].get(organ)
    return None if od is None else od.get(field)


# ----------------------------------------------------------------------------- app
st.set_page_config(page_title="OAKG similarity retrieval", layout="wide")
records = load_records()
rec_by_id = {r["case_id"]: r for r in records}

st.title("Patient similarity retrieval — OAKG vs the paper's baselines")
st.caption(
    "Pick an **anchor patient** in the sidebar via a phenotype range query (OAKG returns only "
    "observation-backed matches). The anchor drives everything below: similarity retrieval against "
    "the OAKG paper's baselines, an interactive KG view, and a Llama 3.2 3B explainer."
)

# ---------------------------------------------------------------- sidebar (all controls)
with st.sidebar:
    st.header("Cohort")
    ds_all = sorted({r["dataset"] for r in records})
    ds_sel = st.multiselect("Datasets in the KG", ds_all, default=ds_all)
    cohort = [r for r in records if r["dataset"] in ds_sel]
    st.caption(f"{len(cohort)} patients · "
               + ", ".join(f"{d}={sum(1 for r in cohort if r['dataset']==d)}" for d in ds_sel))

    st.header("Anchor patient")
    st.caption("Find the anchor with a phenotype range query — OAKG returns only observation-backed "
               "matches; pick one as the anchor for everything on the right.")
    num_key = st.selectbox("Phenotype", list(NUMERIC), format_func=lambda k: NUMERIC[k][0])
    nlabel, norgan, nkind, nfield = NUMERIC[num_key]
    obs_vals = [v for v in (real_value(r, norgan, nfield) for r in cohort
                            if is_observed(r, norgan, nkind)) if v is not None]
    vmax = float(max(obs_vals)) if obs_vals else 1.0

    OPERATORS = {"less than (<)": "<", "at most (≤)": "<=", "greater than (>)": ">",
                 "at least (≥)": ">=", "equals (=)": "==", "between (range)": "between"}
    SYM = {"<": "<", "<=": "≤", ">": ">", ">=": "≥", "==": "="}
    op = OPERATORS[st.selectbox("Condition", list(OPERATORS), index=0, key=f"op_{num_key}")]
    if op == "between":
        lo, hi = st.slider(f"{nlabel} range", 0.0, round(vmax, 1),
                           (0.0, round(vmax * 0.25, 1)), key=f"rng_{num_key}")
        def matches(v): return lo <= v <= hi
        cond_label = f"in [{lo}, {hi}]"
    else:
        thr = st.number_input(f"{nlabel} — threshold", 0.0, round(vmax, 1),
                              0.0 if op == "==" else round(vmax * 0.25, 1), step=0.5,
                              key=f"thr_{num_key}_{op}")
        _ops = {"<": lambda v: v < thr, "<=": lambda v: v <= thr, ">": lambda v: v > thr,
                ">=": lambda v: v >= thr, "==": lambda v: abs(v - thr) < 1e-9}
        matches = _ops[op]
        cond_label = f"{SYM[op]} {thr}"

    cat_key = st.selectbox("Optional categorical filter", ["(none)"] + list(CATEG),
                           format_func=lambda k: k if k == "(none)" else CATEG[k][0])
    cat_allowed = None
    if cat_key != "(none)":
        _, corgan, ckind, cfield = CATEG[cat_key]
        opts = sorted({real_value(r, corgan, cfield) for r in cohort
                       if is_observed(r, corgan, ckind) and real_value(r, corgan, cfield)
                       not in (None, "unknown", "none", "na")})
        cat_allowed = st.multiselect(f"{CATEG[cat_key][0]} in", opts, default=opts[:1] if opts else [])

    def _cat_ok(rec):
        if cat_allowed is None:
            return True
        _, corgan, ckind, cfield = CATEG[cat_key]
        return is_observed(rec, corgan, ckind) and real_value(rec, corgan, cfield) in set(cat_allowed)

    def _oakg_match(rec):
        rv = real_value(rec, norgan, nfield)
        return is_observed(rec, norgan, nkind) and rv is not None and matches(rv) and _cat_ok(rec)

    anchor_matches = sorted((r for r in cohort if _oakg_match(r)),
                            key=lambda r: real_value(r, norgan, nfield), reverse=op in (">", ">="))
    st.caption(f"OAKG matches for `{nlabel} {cond_label}`: **{len(anchor_matches)}** "
               f"(of {len(obs_vals)} that observed this phenotype).")
    if anchor_matches:
        a_labels = [f"{r['case_id']}  ·  {r['dataset']}  ·  {round(real_value(r, norgan, nfield), 2)}"
                    for r in anchor_matches]
        anchor = st.selectbox("Anchor patient", a_labels,
                              key=f"anchor_{num_key}_{cond_label}_{cat_key}").split("  ·  ")[0]
    else:
        anchor = None
        st.warning("No observation-backed match — loosen the condition or dataset filter.")

    st.header("Similarity comparison")
    base = st.selectbox("Paper baseline vs OAKG", list(pr.BASELINE_FUNCTIONS), index=3)  # Masked cosine
    topk = st.selectbox("top-k neighbours", [5, 10, 15, 20], index=1)


# ---------------------------------------------------------------- main: anchor + similarity
if anchor is None:
    st.info("⬅ Pick an **anchor patient** in the sidebar (adjust the range query until OAKG returns "
            "at least one observation-backed match).")
    st.stop()

arec = rec_by_id[anchor]
av = real_value(arec, norgan, nfield)
st.success(f"**Anchor:** `{anchor}`  ·  {arec['dataset']}  ·  observed organs: "
           f"{', '.join(arec['observed_organs'])}  ·  {nlabel} = "
           f"{round(av, 2) if av is not None else '—'}")

st.subheader(f"🔬 Similarity retrieval — OAKG vs {base}")
st.caption("The OAKG paper's exact baselines: rank every other patient against the anchor over a "
           "masked phenotype matrix. OAKG restricts to jointly-observed features and weights by "
           "shared-evidence **γ** (Jaccard of observed organs); baselines don't — so a patient "
           "sharing only one feature can look like a perfect match (low γ).")
corpus, Xm, Mm = load_corpus()
blind = ("⚠️ coverage-blind (fabricates missing values)" if pr.COVERAGE_BLIND[base]
         else "coverage-aware (jointly-observed features only)")
oakg_top = pr.rank(anchor, "OAKG", corpus, Xm, Mm, records, topk)
base_top = pr.rank(anchor, base, corpus, Xm, Mm, records, topk)
low_oakg = sum(t["low evidence (γ<0.25)"] for t in oakg_top)
low_base = sum(t["low evidence (γ<0.25)"] for t in base_top)

k1, k2, k3 = st.columns(3)
k1.metric("Anchor observed organs", ", ".join(sorted(corpus.observation_sets[anchor])) or "—")
k2.metric(f"OAKG weak-overlap (top-{topk})", low_oakg)
k3.metric(f"{base} weak-overlap (top-{topk})", low_base, delta=blind, delta_color="off")


def sim_table(rows_):
    df = pd.DataFrame(rows_)
    df.insert(0, "flag", ["⚠️ weak (γ<0.25)" if r["low evidence (γ<0.25)"] else "✔" for r in rows_])
    return df[["flag", "patient", "dataset", "score", "shared organs", "γ"]]


colL, colR = st.columns(2)
with colL:
    st.markdown(f"### ❌ {base}")
    st.caption(blind)
    st.dataframe(sim_table(base_top), hide_index=True, width="stretch", height=380)
with colR:
    st.markdown("### ✅ OAKG (support-restricted + γ)")
    st.caption("Jointly-observed, anatomically-supported features; γ-weighted so weak-overlap "
               "neighbours sink.")
    st.dataframe(sim_table(oakg_top), hide_index=True, width="stretch", height=380)

if low_base > low_oakg:
    st.error(f"**{base}** put **{low_base}** weak-overlap neighbours (γ<0.25) in its top-{topk} — "
             f"patients sharing almost no observed evidence with `{anchor}`. OAKG has {low_oakg}: its "
             "shared-evidence γ weight pushes those down.")
else:
    st.success(f"On this anchor, {base} and OAKG agree on evidence quality "
               f"(weak-overlap neighbours: {base} {low_base}, OAKG {low_oakg}).")

with st.expander("All paper baselines — weak-overlap neighbours in the top-k (lower = better)"):
    board = [{"method": "🟢 OAKG", "weak-overlap (γ<0.25)": low_oakg, "coverage-blind": "—"}]
    for m in pr.BASELINE_FUNCTIONS:
        rk = pr.rank(anchor, m, corpus, Xm, Mm, records, topk)
        board.append({"method": m, "weak-overlap (γ<0.25)": sum(r["low evidence (γ<0.25)"] for r in rk),
                      "coverage-blind": "yes" if pr.COVERAGE_BLIND[m] else "no (masked)"})
    st.dataframe(pd.DataFrame(board), hide_index=True, width="stretch")
    st.caption("Baselines vendored verbatim from the OAKG paper (`oakg/baselines.py`). Zero/Mean/"
               "Missingness are coverage-blind; Masked cosine & Gower are coverage-aware but "
               "un-weighted, so a tiny overlap still scores high — OAKG's γ is what fixes that.")

# ----------------------------------------------------------------------------- KG visualization
st.divider()
st.subheader("🕸 Knowledge graph — the anchor patient")
import kg_viz
mappings = load_mappings()
st.caption("Interactive vis.js graph of the anchor's KG subgraph — drag nodes, hover for properties, "
           "scroll to zoom. Categorical phenotypes are direct triples, e.g. lesion —tumorBurden→ high.")
st.markdown(" &nbsp; ".join(f"<span style='color:{c};font-size:18px'>●</span> {t}"
                            for t, c in kg_viz.TYPE_COLOR.items()), unsafe_allow_html=True)
components.html(kg_viz.patient_graph_html(arec, mappings, height=560), height=584)
g1, g2 = st.columns(2)
g1.plotly_chart(kg_viz.patient_bars_figure(arec, norgan), width="stretch")
g2.plotly_chart(kg_viz.cohort_bar_figure([rec_by_id[t["patient"]] for t in oakg_top]), width="stretch")
with st.expander("Anchor raw record (KG properties)"):
    st.json(arec)

# ----------------------------------------------------------------------------- Llama 3.2 3B
st.divider()
st.subheader("🧠 Ask Llama 3.2 3B about these results")
st.caption("Grounded in the anchor patient and its OAKG-vs-baseline retrieval. The conversation "
           "**resets when you change the anchor or comparison.**")


@st.cache_resource(show_spinner="Loading Llama 3.2 3B (first use only)…")
def get_llm():
    from llm_backend import load_llama
    return load_llama()


def query_context():
    oakg_ex = ", ".join(f"{t['patient']}({t['dataset']},γ={t['γ']})" for t in oakg_top[:6]) or "none"
    base_ex = ", ".join(f"{t['patient']}({t['dataset']},γ={t['γ']})" for t in base_top[:6]) or "none"
    return (
        f"Anchor patient: {anchor} — dataset {arec['dataset']}, observed organs "
        f"{sorted(arec['observed_organs'])}, {nlabel} = {round(av, 2) if av is not None else 'NA'}.\n"
        "Task: rank other patients by similarity to the anchor over a masked phenotype matrix.\n"
        "OAKG restricts comparison to jointly-OBSERVED features and weights similarity by shared-"
        "evidence gamma (Jaccard of observed organ sets).\n"
        f"Compared baseline: {base} — "
        + ("coverage-blind (fabricates missing values)" if pr.COVERAGE_BLIND[base]
           else "coverage-aware (jointly-observed features only)") + ".\n"
        f"Weak-overlap neighbours (gamma<0.25) in top-{topk}: OAKG={low_oakg}, {base}={low_base}.\n"
        f"OAKG top neighbours (id(dataset,gamma)): {oakg_ex}.\n"
        f"{base} top neighbours: {base_ex}.\n"
        "Key point: a baseline can rank a patient that shares only one observed feature as a perfect "
        "match (cosine of a single scalar = 1.0); OAKG's gamma sinks such weak-overlap matches."
    )


SYSTEM = (
    "You are a data assistant explaining patient-SIMILARITY retrieval results over an imaging "
    "knowledge graph to a clinical research team. Be concise and grounded ONLY in the CONTEXT; do "
    "not invent patients, datasets, or numbers. Key idea: OAKG compares patients only on jointly-"
    "OBSERVED features and weights similarity by shared-evidence gamma (Jaccard of observed organs), "
    "so patients sharing little real evidence with the anchor are down-ranked. Coverage-blind "
    "baselines (zero/mean imputation, missingness indicators) fabricate values for unobserved "
    "features; even coverage-aware ones (masked cosine, Gower) don't weight by gamma, so a tiny "
    "one-feature overlap can score as a perfect match. Lower 'weak-overlap (gamma<0.25)' is better. "
    "Prefer plain clinical language."
)

# reset the chat whenever the anchor or comparison changes
query_sig = (anchor, base, topk, tuple(sorted(ds_sel)))
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
                + f"\n\nWrite a short paragraph for a clinician: who OAKG retrieved as most similar to "
                  f"the anchor and why {base} differs."}]
        st.session_state.chat.append(
            ("assistant", generate(tok, model, msg, max_new_tokens=320, temperature=0.2)))
if b2.button("🗑 Clear chat", width="stretch"):
    st.session_state.chat = []

for role, text in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(text)

prompt = st.chat_input("Ask about the retrieval (e.g. 'why did masked cosine rank FLARE patients?')")
if prompt:
    st.session_state.chat.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"), st.spinner("Llama 3.2 3B is thinking…"):
        from llm_backend import generate
        tok, model, repo = get_llm()
        history = [{"role": r, "content": t} for r, t in st.session_state.chat if r in ("user", "assistant")]
        msgs = [{"role": "system", "content": SYSTEM + "\n\nCONTEXT (current anchor):\n" + query_context()}] + history
        reply = generate(tok, model, msgs, max_new_tokens=350, temperature=0.2)
        st.markdown(reply)
        st.session_state.chat.append(("assistant", reply))
        st.caption(f"powered by {repo}")

st.caption(f"KG source: {os.path.relpath(CORPUS, ROOT)} · {len(records)} patient instances · "
           "observability from `observed_organs` + dataset tumor-annotation coverage.")
