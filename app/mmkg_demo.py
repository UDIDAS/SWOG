#!/usr/bin/env python3
"""MMKG demo — from a new CT to a growing, queryable knowledge graph.

Guided flow on any held-out test patient (our AUSAM pipeline output):
  1 Segment (AUSAM, every slice)  →  2 Phenotype  →  3 Validate vs the train-KG atlas (no labels)  →
  4 Grow the KG (patient becomes an ontology-grounded node)  →
  5 Retrieve N similar patients across the WHOLE KB (OAKG, γ-weighted) with the ontology graph  →
  6 Query the entire KB with the OAKG paper's complex named queries.

Steps 2-6 run GPU-free on precomputed phenotypes; only the step-1 slice viewer runs the models live.

Run:  streamlit run app/mmkg_demo.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
sys.path.insert(0, HERE)
import paper_retrieval as pr   # noqa: E402
import kg_viz                  # noqa: E402  (the vis.js ontology graph)

st.set_page_config(page_title="MMKG demo", layout="wide")
WIN = (-125, 225)
ORGAN_OF = {"msd": "pancreas", "lits": "liver", "kits": "kidney"}
KB_DS = {"msd": "pancreas"}          # our MSD corpus == the KB's 'pancreas' cohort (dataset-label alignment)


# ------------------------------------------------------------------ data (fast, no GPU)
@st.cache_data
def load_patients():
    pats = []
    for ds in ["msd", "lits", "kits"]:
        pf, gf = f"{RES}/corpus_predicted_{ds}.json", f"{RES}/corpus_gt_{ds}.json"
        if not os.path.exists(pf):
            continue
        pred = {r["case_id"]: r for r in json.load(open(pf))["records"]}
        gt = {r["case_id"]: r for r in json.load(open(gf))["records"]} if os.path.exists(gf) else {}
        for cid, rec in pred.items():
            pats.append({"ds": ds, "case": cid, "organ": ORGAN_OF[ds], "pred": rec, "gt": gt.get(cid)})
    return pats


@st.cache_data
def load_kb():
    d = json.load(open(os.path.join(ROOT, "kg", "data", "corpus_perpatient.json")))
    return d["records"] if isinstance(d, dict) else d


@st.cache_data
def load_atlas():
    return json.load(open(os.path.join(ROOT, "kg", "data", "kg_atlas.json")))["organs"]


@st.cache_data
def load_mappings():
    return json.load(open(os.path.join(ROOT, "kg", "ontology_mappings.json")))["mappings"]


PATS, KB, ATLAS, MAPS = load_patients(), load_kb(), load_atlas(), load_mappings()


def kb_case_id(ds, case):     # our ids -> KB naming (LiTS 'volume-93' -> 'LiTs-093'); others match
    return f"LiTs-{int(case.replace('volume-', '')):03d}" if ds == "lits" else case


def _org(r, o):
    return (r or {}).get("organs", {}).get(o, {})


if "admitted" not in st.session_state:
    st.session_state.admitted = {}     # anchor_id -> {"rec":record, "twin":kb_id}; accumulates across session


# ------------------------------------------------------------------ live segmentation (lazy, GPU)
@st.cache_resource(show_spinner="Loading AUSAM models…")
def get_models(ds):
    sys.path.insert(0, os.path.join(ROOT, "src", "scripts"))
    from build_predicted_corpus import CFG, DEV                              # noqa: E402
    from run_pancreas_sam3 import _load_sam3_ckpt, SAM3_MODEL_ID, HF_TOKEN   # noqa: E402
    from transformers import Sam3Processor                                   # noqa: E402
    proc = Sam3Processor.from_pretrained(SAM3_MODEL_ID, token=HF_TOKEN)
    cfg = CFG[ds]
    return proc, _load_sam3_ckpt(cfg["om"], DEV), _load_sam3_ckpt(cfg["tm"], DEV), cfg


@st.cache_data(show_spinner=False)
def segment(ds, case):
    from build_predicted_corpus import predict_volume                        # noqa: E402
    proc, om, tm, cfg = get_models(ds)
    ct, seg, _ = cfg["load"](case)
    omask = predict_volume(om, proc, ct, seg, cfg["olab"], cfg["ax"], cfg["organ"])
    tmask = predict_volume(tm, proc, ct, seg, cfg["tlab"], cfg["ax"], "tumor")
    return ct, omask, tmask, cfg["ax"]


# ------------------------------------------------------------------ complex named queries (OAKG paper set)
def _kidney_tumor(r):
    return any(_org(r, o).get("has_tumor") for o in ("kidney", "right_kidney", "left_kidney"))


QUERIES = {
    "Multi-organ patients (≥2 organs observed)": lambda r: len(r.get("observed_organs", [])) >= 2,
    "High-burden pancreatic tumor": lambda r: _org(r, "pancreas").get("burden_cat") == "high",
    "Contained pancreatic-head tumor": lambda r: (_org(r, "pancreas").get("containment") == "contained"
                                                  and _org(r, "pancreas").get("anatomic_location") == "head"),
    "Multifocal liver tumor": lambda r: _org(r, "liver").get("multiplicity") == "multifocal",
    "Solitary high-burden liver tumor": lambda r: (_org(r, "liver").get("burden_cat") == "high"
                                                   and _org(r, "liver").get("multiplicity") == "solitary"),
    "Kidney-tumor patients (KiTS / FLARE)": _kidney_tumor,
    "Cross-organ tumor (≥2 organs with tumor)": lambda r: sum(1 for o in r.get("observed_organs", [])
                                                              if _org(r, o).get("has_tumor")) >= 2,
    "Any high-burden tumor": lambda r: any(_org(r, o).get("burden_cat") == "high"
                                           for o in r.get("observed_organs", [])),
}

# ------------------------------------------------------------------ header + patient picker
st.title("MMKG — from a new CT to a growing knowledge graph")
st.caption("A held-out patient flows through the pipeline; the knowledge graph validates it, grows by one, and "
           "becomes queryable across every dataset. All segmentation is semi-oracle (interactive-box) AUSAM.")

if not PATS:
    st.error("No predicted corpora found — run `python src/scripts/build_predicted_corpus.py --dataset msd|lits|kits`.")
    st.stop()

st.sidebar.header("Test patient")
ds_pick = st.sidebar.selectbox("Dataset", ["all"] + sorted({p["ds"] for p in PATS}),
                               format_func=lambda d: d.upper())
pool = [p for p in PATS if ds_pick == "all" or p["ds"] == ds_pick]
labels = [f"{p['ds']} · {p['case']} ({p['organ']})" for p in pool]
idx = st.sidebar.selectbox(f"Patient ({len(pool)} held-out)", range(len(pool)), format_func=lambda i: labels[i])
P = pool[idx]
ds, case, organ, pred, gt = P["ds"], P["case"], P["organ"], P["pred"], P["gt"]
st.sidebar.markdown(f"**{len(PATS)}** test patients total  \n**Dataset:** {ds.upper()}  \n**Organ:** {organ}")

# ============================ 1 · Segment (every slice) ============================
st.header("1 · Segment (AUSAM) — scrub every slice")
skey = f"seg::{ds}::{case}"
if st.button("▶ Run AUSAM segmentation for this patient (loads every slice)"):
    st.session_state[skey] = True
if st.session_state.get(skey):
    try:
        with st.spinner(f"Segmenting {case} over the whole volume (first run ~1–3 min, then cached)…"):
            ct, omask, tmask, ax = segment(ds, case)
        Z = ct.shape[ax]
        tumor_sl = [z for z in range(Z) if np.take(tmask, z, ax).any()]
        organ_sl = [z for z in range(Z) if np.take(omask, z, ax).any()]
        c1, c2 = st.columns([3, 2])
        choice = c2.radio("Slice pool", ["Tumor slices", "Organ slices", "All slices"], index=0)
        pool_sl = (tumor_sl if choice == "Tumor slices" and tumor_sl
                   else organ_sl if choice == "Organ slices" and organ_sl else list(range(Z)))
        z = c2.select_slider("Slice", options=pool_sl, value=pool_sl[len(pool_sl) // 2])
        c2.caption(f"**{len(organ_sl)}** organ slices, **{len(tumor_sl)}** tumor slices of {Z} — "
                   f"most slices show no tumor, so the pool defaults to tumor-bearing slices.")
        ctS, oS, tS = np.take(ct, z, ax), np.take(omask, z, ax), np.take(tmask, z, ax)
        lo, hi = WIN
        fig, a = plt.subplots(figsize=(5, 5)); a.imshow(np.clip(ctS, lo, hi), cmap="gray", vmin=lo, vmax=hi)
        if oS.any():
            a.imshow(np.ma.masked_where(~oS, oS.astype(float)), cmap="Greens", alpha=0.35, vmin=0, vmax=1)
        if tS.any():
            a.imshow(np.ma.masked_where(~tS, tS.astype(float)), cmap="autumn", alpha=0.65, vmin=0, vmax=1)
        a.set_title(f"{case} · slice {z} — {organ} (green) + tumor (red)", fontsize=10); a.axis("off")
        c1.pyplot(fig)
    except Exception as e:
        st.error(f"Segmentation unavailable in this session ({type(e).__name__}: {e}). "
                 f"The rest of the pipeline below runs on precomputed phenotypes.")
else:
    st.info("Click **Run AUSAM segmentation** to segment the whole volume and scrub every slice (green organ, "
            "red tumor). Steps 2–6 below already run on the precomputed phenotypes — no GPU needed.")

# ============================ 2 · Phenotype ============================
st.header("2 · Phenotype")
po, go = _org(pred, organ), _org(gt, organ)
rows = [["organ volume (cc)", po.get("organ_volume_cm3"), go.get("organ_volume_cm3", "—")],
        ["tumor volume (cc)", po.get("tumor_volume_cm3"), go.get("tumor_volume_cm3", "—")],
        ["burden", po.get("burden_cat"), go.get("burden_cat", "—")],
        ["multiplicity", po.get("multiplicity"), go.get("multiplicity", "—")],
        ["containment", po.get("containment"), go.get("containment", "—")],
        ["location", po.get("anatomic_location"), go.get("anatomic_location", "—")]]
st.table(pd.DataFrame(rows, columns=["phenotype", "predicted", "ground truth (reference)"]))

# ============================ 3 · Validate ============================
st.header("3 · Validate against the train-KG atlas (no labels)")
band = ATLAS.get(organ, {}).get("volume_cm3", {})
lo_b, hi_b, vol = band.get("p2.5"), band.get("p97.5"), po.get("organ_volume_cm3")
ok = (lo_b is not None and vol is not None and lo_b <= vol <= hi_b)
if ok:
    st.success(f"✅ **Plausible** — predicted {organ} volume {vol} cc is within the cohort band "
               f"[{lo_b:.0f}, {hi_b:.0f}] cc (median {band.get('median','?')}). Admissible to the graph.")
elif lo_b is not None:
    st.error(f"⚠️ **Flagged** — predicted {organ} volume {vol} cc falls outside the plausible band "
             f"[{lo_b:.0f}, {hi_b:.0f}] cc — a likely segmentation error, caught with **no ground truth**.")

# ============================ 4 · Grow the KG ============================
st.header("4 · Grow the knowledge graph")
anchor_id = f"{case}·new"
kb_ds = KB_DS.get(ds, ds)
anchor_rec = json.loads(json.dumps(pred)); anchor_rec["case_id"] = anchor_id; anchor_rec["dataset"] = kb_ds
if st.button(f"➕ Admit {case} to the knowledge graph" + ("" if ok else "  (override — flagged)")):
    st.session_state.admitted[anchor_id] = {"rec": anchor_rec, "twin": kb_case_id(ds, case)}
admitted = dict(st.session_state.admitted)
admitted[anchor_id] = {"rec": anchor_rec, "twin": kb_case_id(ds, case)}   # current anchor always present
twins = {v["twin"] for v in admitted.values()}
KB_grown = [r for r in KB if r["case_id"] not in twins] + [v["rec"] for v in admitted.values()]
st.markdown(f"Train cohort **{len(KB) - len(twins):,}** → **{len(KB) - len(twins) + len(admitted):,}** patients "
            f"(**{len(admitted)}** admitted this session). Each admitted patient is grounded to its organ/tumor "
            f"concepts and becomes queryable across every dataset.")
if len(admitted) > 1:
    with st.expander(f"Ontology view of the {len(admitted)} patients admitted this session"):
        components.html(kg_viz.merged_kg_html([v["rec"] for v in admitted.values()], MAPS, height=420), height=444)

# ============================ 5 · Retrieve similar patients ============================
st.header("5 · Retrieve similar patients across the whole knowledge base")
k = st.slider("N similar patients", 3, 15, 8)
method = st.radio("Method", ["OAKG (γ-weighted)", "Masked cosine"], horizontal=True)
mname = "OAKG" if method.startswith("OAKG") else "Masked cosine"
corpus, X, M = pr.build_corpus(KB_grown)
res = pr.rank(anchor_id, mname, corpus, X, M, KB_grown, k=k)
st.dataframe(pd.DataFrame(res), use_container_width=True)
same = sum(1 for r in res if r["dataset"] == kb_ds)
st.caption(f"{same}/{len(res)} retrieved patients are from the same cohort ({kb_ds}). "
           f"γ = joint observability (shared observed-organ fraction); OAKG never imputes a missing organ, so it "
           f"prefers true same-organ matches (γ=1.0) over thin-overlap ones — masked cosine does not.")

rec_by_id = {r["case_id"]: r for r in KB_grown}
tab1, tab2 = st.tabs(["Retrieval graph (anchor + neighbours)", "Anchor as an ontology-grounded node"])
with tab1:
    graph_recs = [anchor_rec] + [rec_by_id[r["patient"]] for r in res if r["patient"] in rec_by_id]
    components.html(kg_viz.merged_kg_html(graph_recs, MAPS, height=560), height=584)
    st.caption("One merged graph: patients from different datasets connect through the SHARED Dataset and "
               "ontology-Concept nodes — the multi-source integration, visualized.")
with tab2:
    components.html(kg_viz.patient_graph_html(anchor_rec, MAPS, height=560), height=584)
    st.caption("The anchor's own subgraph: Patient → ImagingCase → Organ → Lesion → phenotype observations, each "
               "mapped to a SNOMED CT / NCIt concept.")

# ============================ 6 · Query the whole KB ============================
st.header("6 · Query the entire knowledge base")
q = st.selectbox("Complex query (OAKG paper set)", list(QUERIES))
hits = [r for r in KB_grown if QUERIES[q](r)]
st.markdown(f"**{len(hits):,}** patients match — across {', '.join(sorted({r['dataset'] for r in hits})) or '—'}.")
show = [{"patient": r["case_id"], "dataset": r["dataset"], "organs": ", ".join(r.get("observed_organs", []))}
        for r in hits[:25]]
st.dataframe(pd.DataFrame(show), use_container_width=True)
