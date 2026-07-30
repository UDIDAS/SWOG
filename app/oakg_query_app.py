#!/usr/bin/env python3
"""
OAKG retrieval demos (Streamlit) with a Llama 3.2 3B explainer/chatbot.

Two SEPARATE retrievals, one per tab:

  🔎 Range-based retrieval — structured query (phenotype + condition). OAKG returns only
     observation-backed matches; a coverage-blind competitor imputes missing values and over-returns
     (false positives). Shows the imputation-vs-observability story + a precision leaderboard.

  🧭 Anchor-based similarity — pick any patient as an anchor; rank all others by similarity using the
     OAKG paper's exact baselines (Section 5) vs OAKG's support-restricted + γ-weighted scorer.

Observability is real (from `observed_organs` + dataset tumor annotation): Pancreas patients imaged
only the pancreas, LiTS only the liver, FLARE 5 organ volumes but no tumors.

Run:  streamlit run app/oakg_query_app.py
"""
import json
import os
import re
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
    "panc_contain":("Pancreatic tumor containment",   "pancreas", "tumor", "containment"),
    "panc_loc":    ("Pancreatic tumor location",      "pancreas", "tumor", "anatomic_location"),
    "liver_burden":("Liver tumor burden",             "liver",    "tumor", "burden_cat"),
    "liver_mult":  ("Liver tumor multiplicity",       "liver",    "tumor", "multiplicity"),
    "liver_contain":("Liver tumor containment",       "liver",    "tumor", "containment"),
}
CAT_FIELD_VALUES = {"burden_cat": ["high", "low"], "multiplicity": ["solitary", "multifocal"],
                    "containment": ["contained", "boundary"], "anatomic_location": ["head", "body", "tail"]}
OPERATORS = {"less than (<)": "<", "at most (≤)": "<=", "greater than (>)": ">",
             "at least (≥)": ">=", "equals (=)": "==", "between (range)": "between"}
SYM = {"<": "<", "<=": "≤", ">": ">", ">=": "≥", "==": "="}
TUMOR_ANNOTATED = {("pancreas", "pancreas"), ("liver", "lits")}

# The OAKG paper's structured queries (Table 4), mapped to this app's phenotype schema.
PAPER_STRUCTURED = [
    ("High tumor burden (pancreas)", "panc_tumor_vol", "greater than (>)", 0.0, "panc_burden", ["high"]),
    ("Multifocal disease (liver)", "liver_tumor_vol", "greater than (>)", 0.0, "liver_mult", ["multifocal"]),
    ("Tumor in the pancreas", "panc_tumor_vol", "greater than (>)", 0.0, "(none)", []),
    ("Tumor in the liver", "liver_tumor_vol", "greater than (>)", 0.0, "(none)", []),
    ("Contained pancreatic tumor", "panc_tumor_vol", "greater than (>)", 0.0, "panc_contain", ["contained"]),
    ("Small pancreatic tumor (< 5 cm³)", "panc_tumor_vol", "less than (<)", 5.0, "(none)", []),
]


@st.cache_data
def load_records():
    return json.load(open(CORPUS))["records"]


@st.cache_data
def load_mappings():
    return json.load(open(os.path.join(ROOT, "kg", "ontology_mappings.json")))["mappings"]


@st.cache_resource
def load_corpus():
    return pr.build_corpus(load_records())


@st.cache_data
def load_crossds():
    p = os.path.join(ROOT, "kg", "data", "crossdataset_query_results.json")
    return json.load(open(p))["queries"] if os.path.exists(p) else []


def is_observed(rec, organ, kind):
    if organ not in rec["observed_organs"]:
        return False
    if kind == "organ":
        return True
    return (organ, rec["dataset"]) in TUMOR_ANNOTATED


def real_value(rec, organ, field):
    od = rec["organs"].get(organ)
    return None if od is None else od.get(field)


def naive_value(rec, organ, field, default):
    od = rec["organs"].get(organ)
    if od is None:
        return default
    v = od.get(field)
    return default if v is None else v


# ----------------------------------------------------------------------------- Llama helper
@st.cache_resource(show_spinner="Loading assistant (first use only)…")
def get_llm():
    from llm_backend import load_llama
    return load_llama()


def _gen(chat_key, system, context, user_msg, remember=False):
    if remember:
        st.session_state[chat_key].append(("user", user_msg))
    with st.spinner("Assistant is thinking…"):
        from llm_backend import generate
        tok, model, _ = get_llm()
        hist = [{"role": r, "content": t} for r, t in st.session_state[chat_key]
                if r in ("user", "assistant")]
        base = [{"role": "system", "content": system + "\n\nCONTEXT:\n" + context}]
        msgs = base + (hist if remember else [{"role": "user", "content": user_msg}])
        reply = generate(tok, model, msgs, max_new_tokens=340, temperature=0.2)
    st.session_state[chat_key].append(("assistant", reply))


def _normalize_query(q):
    """Coerce the LLM's JSON into a valid schema spec (3B models are sloppy with keys)."""
    ph = str(q.get("phenotype", "")).strip()
    catf, catv = q.get("categorical_field"), q.get("categorical_values") or []
    op, thr = q.get("operator", ">"), q.get("threshold", 0.0)
    num = ph if ph in NUMERIC else next((k for k in NUMERIC if ph.startswith(k)), None)
    if num is None and ph in CATEG:                      # a categorical key given as the phenotype
        organ = CATEG[ph][1]
        num = {"pancreas": "panc_tumor_vol", "liver": "liver_tumor_vol"}.get(organ)
        catf = catf if catf in CATEG else ph
        if isinstance(thr, str) and not catv:            # threshold like "high" is really a value
            catv = [thr]
        op, thr = ">", 0.0
    if num is None:
        return None
    if op not in ("<", "<=", ">", ">=", "==", "between"):
        op = ">"
    if op == "between":                                  # we fill a single threshold, not a range
        op = "<="
    try:
        thr = float(thr)
    except (TypeError, ValueError):
        thr = 0.0
    catf = catf if catf in CATEG else None
    if catf:
        allowed = CAT_FIELD_VALUES.get(CATEG[catf][3], [])
        catv = [v for v in catv if v in allowed]
        if not catv:
            catf = None
    return {"phenotype": num, "operator": op, "threshold": thr,
            "categorical_field": catf, "categorical_values": catv}


def nl_to_query(desc):
    """Use Llama to map a free-text description to a structured phenotype query (JSON)."""
    num_s = "; ".join(f'"{k}" ({NUMERIC[k][0]})' for k in NUMERIC)
    cat_s = "; ".join(f'"{k}" (values {CAT_FIELD_VALUES.get(CATEG[k][3], [])})' for k in CATEG)
    sysp = ("You convert a clinician's description into ONE structured phenotype query. Reply with "
            "ONLY compact JSON and nothing else: {\"phenotype\":\"<numeric key>\",\"operator\":\"<one "
            "of <,<=,>,>=,==,between>\",\"threshold\":<number>,\"categorical_field\":\"<categorical "
            "key or null>\",\"categorical_values\":[<strings>]}. "
            f"Numeric keys (use EXACTLY one, no suffix): {num_s}. Categorical keys: {cat_s}. "
            "Rules: 'has a tumor' -> the matching tumor-volume key, operator '>', threshold 0. "
            "'small/low' -> operator '<' with a small threshold; 'large/big' -> operator '>'. "
            "Burden/multiplicity/containment/location go in categorical_field + categorical_values, "
            "NOT in phenotype. Use ONLY the listed keys and values.")
    from llm_backend import generate
    tok, model, _ = get_llm()
    raw = generate(tok, model, [{"role": "system", "content": sysp},
                                {"role": "user", "content": f'Description: "{desc}"'}],
                   max_new_tokens=180, temperature=0.1)
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None, raw
    try:
        return _normalize_query(json.loads(m.group(0))), raw
    except Exception:
        return None, raw


def make_condition(op, thr):
    return {"<": lambda v: v < thr, "<=": lambda v: v <= thr, ">": lambda v: v > thr,
            ">=": lambda v: v >= thr, "==": lambda v: abs(v - thr) < 1e-9}.get(op, lambda v: v > thr)


def oakg_range_hits(recs, num_key, matches_fn, cat_key, cat_vals):
    """Observation-backed patients whose phenotype satisfies the condition (+ optional categorical)."""
    _, no, nk, nf = NUMERIC[num_key]
    hits = []
    for r in recs:
        rv = real_value(r, no, nf)
        if not (is_observed(r, no, nk) and rv is not None and matches_fn(rv)):
            continue
        if cat_key and cat_key != "(none)":
            _, co, ck, cf = CATEG[cat_key]
            if not (is_observed(r, co, ck) and real_value(r, co, cf) in set(cat_vals)):
                continue
        hits.append(r)
    return hits


def blind_count(recs, num_key, matches_fn, cat_key, cat_vals):
    """How many a coverage-blind (impute-0 / 'none') reader would return."""
    _, no, nk, nf = NUMERIC[num_key]
    n = 0
    for r in recs:
        if not matches_fn(naive_value(r, no, nf, 0.0)):
            continue
        if cat_key and cat_key != "(none)" \
                and naive_value(r, CATEG[cat_key][1], CATEG[cat_key][3], "none") not in set(cat_vals):
            continue
        n += 1
    return n


def apply_nl_query(num, op_label, thr, cat, catv):
    """Fill the Range tab controls AND narrow their options to the query's organ scope."""
    scope = NUMERIC[num][1]
    st.session_state.r_scope = scope
    st.session_state.r_ph = num
    st.session_state.r_op = op_label
    st.session_state.r_thr = float(thr if thr is not None else 0.0)
    st.session_state.r_cat = cat if cat in CATEG else "(none)"
    st.session_state.r_catv = list(catv or [])
    st.session_state.r_ds = [d for d in sorted(DATASET_ORGANS) if scope in DATASET_ORGANS[d]]
    st.rerun()


def llm_block(context, system, sig, key, placeholder):
    """Reusable assistant explainer + chat (form-based). sig=None disables auto-reset."""
    sig_key, chat_key = f"{key}_sig", f"{key}_chat"
    if sig is not None and st.session_state.get(sig_key) != sig:     # new query -> fresh chat
        st.session_state[chat_key] = []
        st.session_state[sig_key] = sig
    st.session_state.setdefault(chat_key, [])
    c1, c2 = st.columns(2)
    if c1.button("📝 Summarize the current panels", key=f"{key}_ex", width="stretch"):
        _gen(chat_key, system, context,
             "Summarize the current results for a clinician in a short paragraph.")
    if c2.button("🗑 Clear chat", key=f"{key}_cl", width="stretch"):
        st.session_state[chat_key] = []
    with st.form(key=f"{key}_form", clear_on_submit=True):
        q = st.text_input("Ask", placeholder=placeholder, key=f"{key}_q",
                          label_visibility="collapsed")
        if st.form_submit_button("Ask") and q.strip():
            _gen(chat_key, system, context, q.strip(), remember=True)
    for role, text in st.session_state[chat_key]:               # render AFTER processing
        with st.chat_message(role):
            st.markdown(text)


# ----------------------------------------------------------------------------- app
st.set_page_config(page_title="OAKG retrieval demos", layout="wide")
records = load_records()
rec_by_id = {r["case_id"]: r for r in records}
ds_all = sorted({r["dataset"] for r in records})
DATASET_ORGANS = {}                                    # dataset -> set of organs it observes
for _r in records:
    DATASET_ORGANS.setdefault(_r["dataset"], set()).update(_r["observed_organs"])

st.title("OAKG retrieval demos")
st.caption("Three panels — **Range query** (OAKG vs coverage-blind imputation), **Similar patients** "
           "(OAKG vs the paper's baselines), and **Paper queries** — with an assistant at the bottom.")

CTX = {}   # each panel records a short context string; the bottom assistant reads all of them
tab_range, tab_anchor, tab_nl = st.tabs(
    ["🔎 Range query", "🧭 Similar patients", "📄 Paper queries"])

# ==================================================================== TAB 1: range-based
with tab_range:
    st.caption("Find patients whose phenotype satisfies a condition — set it directly below, or "
               "**describe it in natural language**. OAKG returns only observation-backed matches; the "
               "competitor fabricates missing values and over-returns.")
    for _k, _v in {"r_ph": "panc_tumor_vol", "r_op": "less than (<)", "r_thr": 5.0,
                   "r_cat": "(none)", "r_catv": [], "r_scope": None, "r_ds": list(ds_all)}.items():
        st.session_state.setdefault(_k, _v)

    with st.expander("🗣 Describe the patients in natural language", expanded=bool(st.session_state.r_scope)):
        dc1, dc2 = st.columns([4, 1])
        desc = dc1.text_input("Description", key="r_desc", label_visibility="collapsed",
                              placeholder="e.g. small pancreatic tumors that are contained")
        if dc2.button("💡 Suggest", key="r_suggest", width="stretch") and desc.strip():
            with st.spinner("Assistant is mapping your description…"):
                q, raw = nl_to_query(desc.strip())
            if q:
                op_lab = next((l for l, c in OPERATORS.items() if c == q.get("operator")),
                              "greater than (>)")
                cat = q.get("categorical_field")
                apply_nl_query(q["phenotype"], op_lab, q.get("threshold", 0.0),
                               cat if cat in CATEG else "(none)", q.get("categorical_values") or [])
            else:
                st.warning(f"Could not parse a query. The assistant said: {raw[:200]}")
        st.caption("The paper's structured & cross-dataset queries are in the 📄 Paper queries tab.")

    # natural-language scope narrows the dropdown OPTIONS to the query's organ (else show all)
    scope = st.session_state.r_scope
    ph_opts = [k for k in NUMERIC if scope is None or NUMERIC[k][1] == scope]
    ds_opts = [d for d in ds_all if scope is None or scope in DATASET_ORGANS.get(d, set())]
    cat_opts = ["(none)"] + [k for k in CATEG if scope is None or CATEG[k][1] == scope]
    if st.session_state.r_ph not in ph_opts:
        st.session_state.r_ph = ph_opts[0]
    if st.session_state.r_cat not in cat_opts:
        st.session_state.r_cat = "(none)"
    st.session_state.r_ds = [d for d in st.session_state.r_ds if d in ds_opts] or list(ds_opts)
    if scope:
        sc1, sc2 = st.columns([3, 1])
        sc1.caption(f"🔒 Options limited to the **{scope.replace('_', ' ')}** (from your description).")
        if sc2.button("🔓 Show all options", width="stretch"):
            st.session_state.r_scope = None
            st.session_state.r_ds = list(ds_all)
            st.rerun()

    r1 = st.columns([1.2, 1.4, 1.2, 1.2])
    ds_sel = r1[0].multiselect("Datasets", ds_opts, key="r_ds")
    cohort = [r for r in records if r["dataset"] in ds_sel]
    num_key = r1[1].selectbox("Phenotype", ph_opts, format_func=lambda k: NUMERIC[k][0], key="r_ph")
    nlabel, norgan, nkind, nfield = NUMERIC[num_key]
    obs_vals = [v for v in (real_value(r, norgan, nfield) for r in cohort
                            if is_observed(r, norgan, nkind)) if v is not None]
    vmax = float(max(obs_vals)) if obs_vals else 1.0
    op = OPERATORS[r1[2].selectbox("Condition", list(OPERATORS), key="r_op")]
    if op == "between":
        lo, hi = r1[3].slider("range", 0.0, round(vmax, 1), (0.0, round(vmax * 0.25, 1)),
                              key="r_rng", label_visibility="collapsed")
        def matches(v): return lo <= v <= hi
        target, cond_label = (lo + hi) / 2, f"in [{lo}, {hi}]"
    else:
        st.session_state.r_thr = min(float(st.session_state.r_thr), round(vmax, 1))
        thr = r1[3].number_input("threshold", 0.0, round(vmax, 1), step=0.5, key="r_thr")
        matches = make_condition(op, thr)
        target, cond_label = {"<": 0.0, "<=": 0.0, ">": vmax, ">=": vmax, "==": thr}[op], f"{SYM[op]} {thr}"

    r2 = st.columns([1.4, 1.6, 1.0])
    cat_key = r2[0].selectbox("Categorical filter", cat_opts,
                              format_func=lambda k: k if k == "(none)" else CATEG[k][0], key="r_cat")
    cat_allowed = None
    if cat_key != "(none)":
        _, corgan, ckind, cfield = CATEG[cat_key]
        opts = sorted({real_value(r, corgan, cfield) for r in cohort
                       if is_observed(r, corgan, ckind) and real_value(r, corgan, cfield)
                       not in (None, "unknown", "none", "na")})
        st.session_state.r_catv = [v for v in st.session_state.r_catv if v in opts] \
            or (opts[:1] if opts else [])
        cat_allowed = r2[0].multiselect(f"{CATEG[cat_key][0]} in", opts, key="r_catv")
    mean_v = round(statistics.mean(obs_vals), 2) if obs_vals else 0.0
    median_v = round(statistics.median(obs_vals), 2) if obs_vals else 0.0
    COMPETITORS = {
        "Zero imputation (missing → 0)": ("impute", 0.0, "unobserved value filled with 0"),
        "Mean imputation (missing → cohort mean)":
            ("impute", mean_v, f"unobserved value filled with the cohort mean ({mean_v})"),
        "Median imputation (missing → cohort median)":
            ("impute", median_v, f"unobserved value filled with the cohort median ({median_v})"),
        "Cross-organ collision (untyped phenotype)":
            ("cross_organ", None, "any organ's value answers the query — right number, wrong organ"),
    }
    comp_label = r2[1].selectbox("Competitor (vs OAKG)", list(COMPETITORS), index=0, key="r_comp")
    comp_code, comp_const, comp_src = COMPETITORS[comp_label]
    comp_short = comp_label.split(" (")[0]
    n_rows = r2[2].selectbox("Rows / panel", [5, 10, 15, 20, 30, 50], index=1, key="r_rows")

    def relevance(v):
        return 0.0 if v is None else round(max(0.0, 1 - abs(v - target) / (vmax + 1e-9)), 3)

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
        return is_observed(rec, norgan, nkind) and rv is not None and matches(rv) and cat_ok_oakg(rec)

    def comp_match(rec, code, const):
        if code == "cross_organ":
            vals = [od.get(nfield) for od in rec["organs"].values() if od.get(nfield) is not None]
            num = any(matches(v) for v in vals)
        else:
            num = matches(naive_value(rec, norgan, nfield, const))
        return num and cat_ok_comp(rec)

    def comp_seen(rec, code, const):
        if code == "cross_organ":
            vals = [od.get(nfield) for od in rec["organs"].values() if od.get(nfield) is not None]
            return next((v for v in vals if matches(v)), vals[0] if vals else 0.0)
        return naive_value(rec, norgan, nfield, const)

    oakg_ids = {r["case_id"] for r in cohort if oakg_match(r)}

    def row_of(rec):
        rv = real_value(rec, norgan, nfield)
        seen = comp_seen(rec, comp_code, comp_const)
        oak = rec["case_id"] in oakg_ids
        comp = comp_match(rec, comp_code, comp_const)
        sn = seen if isinstance(seen, (int, float)) else None
        return {"patient": rec["case_id"], "dataset": rec["dataset"],
                "real": (round(rv, 2) if rv is not None else None),
                "seen": (round(seen, 2) if sn is not None else seen),
                "observed": "observed" if is_observed(rec, norgan, nkind) else "UNOBSERVED",
                "relevance": relevance(rv if oak else sn),
                "oakg": oak, "comp": comp, "false": comp and not oak}

    rows = [row_of(r) for r in cohort]
    comp_rows = [r for r in rows if r["comp"]]
    oakg_rows = [r for r in rows if r["oakg"]]
    false_rows = [r for r in rows if r["false"]]

    leader = [{"method": "🟢 OAKG (guardrail)", "returned": len(oakg_ids), "false positives": 0,
               "precision": 1.0}]
    for lab, (code, const, src) in COMPETITORS.items():
        ret = {r["case_id"] for r in cohort if comp_match(r, code, const)}
        prec = round(len(ret & oakg_ids) / len(ret), 3) if ret else 1.0
        leader.append({"method": ("▶ " if lab == comp_label else "") + lab.split(" (")[0],
                       "returned": len(ret), "false positives": len(ret - oakg_ids), "precision": prec})

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cohort", len(cohort))
    m2.metric("Observed for phenotype", len(obs_vals))
    m3.metric(f"{comp_short} matches", len(comp_rows),
              delta=f"{len(false_rows)} false", delta_color="inverse")
    m4.metric("OAKG matches (reliable)", len(oakg_rows))
    if false_rows:
        st.error(f"⚠️ **{comp_short}** returned **{len(false_rows)}** false positives for "
                 f"'{nlabel} {cond_label}' — not observation-backed ({comp_src}). OAKG excludes them.")
    else:
        st.success(f"**{comp_short}** produced no false positives for '{nlabel} {cond_label}' here — "
                   "its fabricated value doesn't satisfy the condition. Try 'less than' / '= 0'.")

    with st.expander("Retrieval-quality leaderboard — OAKG vs every competitor", expanded=True):
        st.dataframe(pd.DataFrame(leader).sort_values("precision", ascending=False),
                     hide_index=True, width="stretch")
        st.caption("**Precision** = fraction of a method's returns that are observation-backed (agree "
                   "with OAKG). OAKG is the reference at 1.00.")

    cL, cR = st.columns(2)
    with cL:
        st.markdown(f"### ❌ {comp_short}")
        st.caption(f"{comp_src}. ⚠️ = false positive (not observation-backed).")
        if comp_rows:
            df = pd.DataFrame(comp_rows).sort_values("relevance", ascending=False).head(n_rows)
            df.insert(0, "flag", ["⚠️ false" if r["false"] else "✔" for r in df.to_dict("records")])
            st.dataframe(df.rename(columns={"seen": "value used"})
                         [["flag", "patient", "dataset", "value used", "relevance", "observed"]],
                         hide_index=True, width="stretch", height=360)
        else:
            st.info("No matches.")
    with cR:
        st.markdown("### ✅ OAKG (missing → unobserved)")
        st.caption("Only patients that actually observed the phenotype — no fabricated matches.")
        if oakg_rows:
            df = pd.DataFrame(oakg_rows).sort_values("relevance", ascending=False).head(n_rows)
            st.dataframe(df.rename(columns={"real": nlabel})[["patient", "dataset", nlabel, "relevance"]],
                         hide_index=True, width="stretch", height=360)
        else:
            st.info("No observation-backed patient satisfies this query.")

    if oakg_rows:
        import kg_viz
        st.markdown("**Retrieved patients as one KG** — they connect through shared **Dataset** and "
                    "**SNOMED/NCIt concept** nodes (the shared schema). Drag / hover / zoom.")
        gsz = st.selectbox("patients to graph (top by relevance)", [5, 8, 12, 20], index=1, key="r_gsz")
        st.markdown(" &nbsp; ".join(f"<span style='color:{c};font-size:16px'>●</span> {t}"
                                    for t, c in kg_viz.TYPE_COLOR.items()), unsafe_allow_html=True)
        top = sorted(oakg_rows, key=lambda r: r["relevance"], reverse=True)[:gsz]
        components.html(kg_viz.merged_kg_html([rec_by_id[r["patient"]] for r in top],
                                              load_mappings(), height=560), height=584)
        st.plotly_chart(kg_viz.cohort_bar_figure([rec_by_id[r["patient"]] for r in oakg_rows]),
                        width="stretch")

    CTX["range"] = (
        f"Range query: {nlabel} {cond_label} (categorical filter: "
        f"{CATEG[cat_key][0]+' in '+str(cat_allowed) if cat_allowed else 'none'}).\n"
        f"Cohort {len(cohort)}; {len(obs_vals)} observed this phenotype.\n"
        f"OAKG (observability guardrail) returned {len(oakg_rows)} reliable patients.\n"
        f"Competitor {comp_label} ({comp_src}) returned {len(comp_rows)}, {len(false_rows)} FALSE "
        f"positives.\nLeaderboard: "
        + " | ".join(f"{d['method']}: returned={d['returned']}, false={d['false positives']}, "
                     f"precision={d['precision']}" for d in leader))

# ==================================================================== TAB 2: anchor-based
with tab_anchor:
    st.caption("Pick any patient as the anchor; rank all others by similarity using the OAKG paper's "
               "baselines. OAKG restricts to jointly-observed features and weights by shared-evidence "
               "**γ** (Jaccard of observed organs); baselines don't.")
    corpus, Xm, Mm = load_corpus()
    a_labels = [f"{r['case_id']}  ·  {r['dataset']}" for r in records]
    a1, a2, a3 = st.columns([2.4, 1.6, 1.0])
    anchor = a1.selectbox("Anchor patient", a_labels, key="a_anchor").split("  ·  ")[0]
    base = a2.selectbox("Paper baseline vs OAKG", list(pr.BASELINE_FUNCTIONS), index=3, key="a_base")
    topk = a3.selectbox("top-k", [5, 10, 15, 20], index=1, key="a_topk")
    arec = rec_by_id[anchor]

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

    cL, cR = st.columns(2)
    with cL:
        st.markdown(f"### ❌ {base}")
        st.caption(blind)
        st.dataframe(sim_table(base_top), hide_index=True, width="stretch", height=360)
    with cR:
        st.markdown("### ✅ OAKG (support-restricted + γ)")
        st.caption("γ-weighted so weak-overlap neighbours sink.")
        st.dataframe(sim_table(oakg_top), hide_index=True, width="stretch", height=360)

    if low_base > low_oakg:
        st.error(f"**{base}** put **{low_base}** weak-overlap neighbours (γ<0.25) in its top-{topk} — "
                 f"patients sharing almost no observed evidence with `{anchor}`. OAKG has {low_oakg}.")
    else:
        st.success(f"On this anchor, {base} and OAKG agree on evidence quality "
                   f"(weak-overlap: {base} {low_base}, OAKG {low_oakg}).")

    with st.expander("All paper baselines — weak-overlap neighbours in the top-k (lower = better)"):
        board = [{"method": "🟢 OAKG", "weak-overlap (γ<0.25)": low_oakg, "coverage-blind": "—"}]
        for m in pr.BASELINE_FUNCTIONS:
            rk = pr.rank(anchor, m, corpus, Xm, Mm, records, topk)
            board.append({"method": m,
                          "weak-overlap (γ<0.25)": sum(r["low evidence (γ<0.25)"] for r in rk),
                          "coverage-blind": "yes" if pr.COVERAGE_BLIND[m] else "no (masked)"})
        st.dataframe(pd.DataFrame(board), hide_index=True, width="stretch")
        st.caption("Baselines vendored verbatim from the OAKG paper (`oakg/baselines.py`).")

    st.divider()
    st.markdown("### 🕸 Knowledge graph")
    import kg_viz
    mappings = load_mappings()
    view = st.radio("View", ["Merged KG (anchor + neighbours)", "Single patient (anchor)"],
                    horizontal=True, key="a_view")
    st.markdown(" &nbsp; ".join(f"<span style='color:{c};font-size:16px'>●</span> {t}"
                                for t, c in kg_viz.TYPE_COLOR.items()), unsafe_allow_html=True)
    if view.startswith("Merged"):
        st.caption("Anchor + its OAKG neighbours as **one** KG — connected through shared Dataset and "
                   "SNOMED/NCIt concept nodes. Drag / hover / zoom.")
        merged = [arec] + [rec_by_id[t["patient"]] for t in oakg_top]
        components.html(kg_viz.merged_kg_html(merged, mappings, height=560), height=584)
    else:
        st.caption("The anchor's own subgraph — drag nodes, hover, zoom. Categorical phenotypes are "
                   "direct triples, e.g. lesion —tumorBurden→ high.")
        components.html(kg_viz.patient_graph_html(arec, mappings, height=560), height=584)
    aorgan = {"pancreas": "pancreas", "lits": "liver"}.get(arec["dataset"], arec["observed_organs"][0])
    g1, g2 = st.columns(2)
    g1.plotly_chart(kg_viz.patient_bars_figure(arec, aorgan), width="stretch")
    g2.plotly_chart(kg_viz.cohort_bar_figure([rec_by_id[t["patient"]] for t in oakg_top]),
                    width="stretch")
    with st.expander("Anchor raw record (KG properties)"):
        st.json(arec)

    oakg_ex = ", ".join(f"{t['patient']}({t['dataset']},γ={t['γ']})" for t in oakg_top[:6]) or "none"
    base_ex = ", ".join(f"{t['patient']}({t['dataset']},γ={t['γ']})" for t in base_top[:6]) or "none"
    CTX["anchor"] = (
        f"Anchor: {anchor} (dataset {arec['dataset']}, observed organs {sorted(arec['observed_organs'])}).\n"
        "Task: rank other patients by similarity; OAKG uses jointly-observed features weighted by "
        "shared-evidence gamma (Jaccard of observed organs).\n"
        f"Compared baseline: {base} — "
        + ("coverage-blind" if pr.COVERAGE_BLIND[base] else "coverage-aware (jointly-observed only)")
        + f".\nWeak-overlap neighbours (gamma<0.25) in top-{topk}: OAKG={low_oakg}, {base}={low_base}.\n"
        f"OAKG top neighbours: {oakg_ex}.\n{base} top neighbours: {base_ex}.\n"
        "A baseline can rank a patient sharing only one observed feature as a perfect match "
        "(cosine of a single scalar = 1.0); OAKG's gamma sinks such weak-overlap matches.")

# ==================================================================== TAB 3: paper cross-dataset queries
with tab_nl:
    st.caption("The OAKG paper's queries — **run here**. Structured queries run OAKG's observation-"
               "backed retrieval; cross-dataset queries run OAKG-vs-baseline similarity from a query "
               "patient.")

    st.markdown("#### Structured queries (Table 4)")
    st.caption("Boolean phenotype queries over the KG. Click one to run it.")
    st.session_state.setdefault("paper_struct", 0)
    pcols = st.columns(3)
    for i, spec in enumerate(PAPER_STRUCTURED):
        if pcols[i % 3].button(spec[0], key=f"ps_{i}", width="stretch"):
            st.session_state.paper_struct = i
    st.caption("*Cross-organ distribution* (tumor in ≥2 organs) is **indeterminate** here — every case "
               "observes a single organ, so OAKG returns 'unknown' rather than a false answer.")

    _spec = PAPER_STRUCTURED[st.session_state.paper_struct]
    _nk, _op, _thr, _catk, _catv = _spec[1], OPERATORS[_spec[2]], _spec[3], _spec[4], _spec[5]
    _mfn = make_condition(_op, _thr)
    _hits = oakg_range_hits(records, _nk, _mfn, _catk, _catv)
    _bl = blind_count(records, _nk, _mfn, _catk, _catv)
    st.markdown(f"**Running _{_spec[0]}_** → `{NUMERIC[_nk][0]} {SYM[_op]} {_thr}`"
                + (f"  and  {CATEG[_catk][0]} ∈ {_catv}" if _catk != "(none)" else ""))
    sm1, sm2 = st.columns(2)
    sm1.metric("OAKG matches (observation-backed)", len(_hits))
    sm2.metric("Coverage-blind (impute 0) would return", _bl,
               delta=f"{_bl-len(_hits)} false", delta_color="inverse")
    if _hits:
        st.dataframe(pd.DataFrame([{"patient": r["case_id"], "dataset": r["dataset"],
                                    NUMERIC[_nk][0]: round(real_value(r, NUMERIC[_nk][1],
                                                                      NUMERIC[_nk][3]), 2)}
                                   for r in _hits]).head(50), hide_index=True, width="stretch", height=240)
        import kg_viz
        components.html(kg_viz.merged_kg_html(_hits[:8], load_mappings(), height=440), height=464)

    st.divider()
    st.markdown("#### Cross-dataset queries (B1–B7)")
    st.markdown(
        "Each one **starts from a single patient** and looks for similar patients in a **different "
        "dataset** that share an organ — testing whether the shared schema lets OAKG match *across* "
        "datasets that image different organs (e.g. a Pancreas case vs FLARE cases — both observe the "
        "pancreas). Click **Run** to score OAKG vs Masked cosine for that patient.")
    cds = load_crossds()
    st.session_state.setdefault("paper_anchor", "")
    for q in cds:
        query = q.get("query", {})
        cid = query.get("case_id", "")
        organ = q.get("shared_organ", "")
        od = query.get(organ, {}) if isinstance(query.get(organ), dict) else {}
        feats = [f for f in [od.get("burden_cat") and f"{od['burden_cat']}-burden",
                             od.get("multiplicity"), od.get("containment")] if f]
        pheno = ", ".join(feats) if feats else "a tumor"
        loadable = cid in rec_by_id
        c1, c2 = st.columns([6, 1])
        c1.markdown(
            f"**{q['code']}** — from `{cid}` ({query.get('dataset', '?')}): a **{pheno}** "
            f"{organ.replace('_', ' ')} tumor → find similar patients in "
            f"**{q.get('target_dataset', '?')}** that share the {organ.replace('_', ' ')}."
            + ("" if loadable else "  \n*(slice-level FLARE query — not in this patient-level demo)*"))
        if c2.button("Run", key=f"bq_{q['code']}", width="stretch", disabled=not loadable):
            st.session_state.paper_anchor = cid

    if st.session_state.paper_anchor in rec_by_id:
        _ac = st.session_state.paper_anchor
        _corpus, _Xm, _Mm = load_corpus()
        _otop = pr.rank(_ac, "OAKG", _corpus, _Xm, _Mm, records, 10)
        _btop = pr.rank(_ac, "Masked cosine", _corpus, _Xm, _Mm, records, 10)
        _lo = sum(t["low evidence (γ<0.25)"] for t in _otop)
        _lb = sum(t["low evidence (γ<0.25)"] for t in _btop)
        st.markdown(f"**Most similar to `{_ac}`** — OAKG vs Masked cosine "
                    f"(weak-overlap γ<0.25: OAKG {_lo}, Masked cosine {_lb}):")

        def _simtab(rows_):
            df = pd.DataFrame(rows_)
            df.insert(0, "flag", ["⚠️ weak" if r["low evidence (γ<0.25)"] else "✔" for r in rows_])
            return df[["flag", "patient", "dataset", "score", "shared organs", "γ"]]
        xc1, xc2 = st.columns(2)
        xc1.markdown("**❌ Masked cosine**")
        xc1.dataframe(_simtab(_btop), hide_index=True, width="stretch", height=300)
        xc2.markdown("**✅ OAKG (support-restricted + γ)**")
        xc2.dataframe(_simtab(_otop), hide_index=True, width="stretch", height=300)
    CTX["crossds"] = ("Cross-dataset paper queries B1–B7 (similarity from a query patient to another "
                      "dataset sharing an organ): " + ", ".join(f"{q['code']} {q['title']}" for q in cds)
                      if cds else "none")

# ==================================================================== bottom: assistant
st.divider()
st.subheader("💬 Assistant")
st.caption("Ask about any panel above — the assistant sees the current range, anchor, and describe "
           "queries and their results.")
universal_ctx = "\n\n".join([
    "RANGE-BASED PANEL —\n" + CTX.get("range", "(not run)"),
    "ANCHOR-BASED PANEL —\n" + CTX.get("anchor", "(not run)"),
    "CROSS-DATASET PAPER QUERIES —\n" + CTX.get("crossds", "(none)"),
])
ASSIST_SYS = (
    "You are an assistant for an OAKG retrieval dashboard with three panels: range-based (OAKG vs "
    "coverage-blind imputation), anchor-based similarity (OAKG vs the paper's baselines, weighted by "
    "shared-evidence gamma), and a describe/paper-query panel. Answer grounded ONLY in the CONTEXT of "
    "the current panel states; be concise; don't invent patients or numbers. Core idea: OAKG treats "
    "unobserved phenotypes as unknown and weights similarity by shared evidence, avoiding the false "
    "positives that imputation or unweighted similarity produce.")
llm_block(universal_ctx, ASSIST_SYS, None, "assistant", "Ask about any panel…")

st.caption(f"KG source: {os.path.relpath(CORPUS, ROOT)} · {len(records)} patient instances · "
           "observability from `observed_organs` + dataset tumor-annotation coverage.")
