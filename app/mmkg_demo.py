#!/usr/bin/env python3
"""MMKG demo — from a new CT to a growing, queryable knowledge graph.

Guided flow on a held-out test patient (our AUSAM pipeline output):
  1 Segment (AUSAM)  →  2 Phenotype  →  3 Validate vs the train-KG atlas (no labels)  →
  4 Grow the KG (patient becomes a node)  →  5 Retrieve N similar patients across the WHOLE KB (OAKG, γ-weighted)
  6 Query the entire KB with the OAKG paper's complex named queries.

Run:  streamlit run app/mmkg_demo.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import paper_retrieval as pr   # noqa: E402

st.set_page_config(page_title="MMKG demo", layout="wide")


@st.cache_data
def load_kb():
    d = json.load(open(os.path.join(ROOT, "kg", "data", "corpus_perpatient.json")))
    return d["records"] if isinstance(d, dict) else d


@st.cache_data
def load_atlas():
    return json.load(open(os.path.join(ROOT, "kg", "data", "kg_atlas.json")))["organs"]


@st.cache_data
def load_demo():
    p = os.path.join(HERE, "assets", "demo", "demo_cases.json")
    return json.load(open(p)) if os.path.exists(p) else []


KB, ATLAS, DEMO = load_kb(), load_atlas(), load_demo()

# ---- complex named queries (from the OAKG paper), as record filters ----
def _org(r, o):
    return r.get("organs", {}).get(o, {})


QUERIES = {
    "Multi-organ patients (≥2 organs observed)": lambda r: len(r.get("observed_organs", [])) >= 2,
    "Patients observing pancreas": lambda r: "pancreas" in r.get("observed_organs", []),
    "High-burden pancreatic tumor": lambda r: _org(r, "pancreas").get("burden_cat") == "high",
    "Contained pancreatic-head tumor": lambda r: (_org(r, "pancreas").get("containment") == "contained"
                                                  and _org(r, "pancreas").get("anatomic_location") == "head"),
    "Multifocal liver tumor": lambda r: _org(r, "liver").get("multiplicity") == "multifocal",
    "Solitary high-burden liver tumor": lambda r: (_org(r, "liver").get("burden_cat") == "high"
                                                   and _org(r, "liver").get("multiplicity") == "solitary"),
    "Cross-organ tumor (≥2 organs with tumor)": lambda r: sum(1 for o in r.get("observed_organs", [])
                                                              if _org(r, o).get("has_tumor")) >= 2,
    "Any high-burden tumor": lambda r: any(_org(r, o).get("burden_cat") == "high"
                                           for o in r.get("observed_organs", [])),
}

st.title("MMKG — from a new CT to a growing knowledge graph")
st.caption("A held-out patient flows through the pipeline; the knowledge graph validates it, grows by one, and "
           "becomes queryable across every dataset. All segmentation is semi-oracle (interactive-box) AUSAM.")

if not DEMO:
    st.error("No demo assets yet — run `python src/scripts/build_demo_assets.py` to generate them.")
    st.stop()

# ---- pick a test case ----
labels = [f"{d['dataset']} · {d['case']} ({d['organ']})" for d in DEMO]
idx = st.sidebar.selectbox("Choose a held-out test patient", range(len(DEMO)), format_func=lambda i: labels[i])
D = DEMO[idx]
pred, gt = D["pred"], D.get("gt")
organ = D["organ"]
st.sidebar.markdown(f"**Dataset:** {D['dataset']}  \n**Organ:** {organ}")

# ============================ 1 · Segment ============================
st.header("1 · Segment (AUSAM)")
c1, c2 = st.columns([1, 1])
png = os.path.join(HERE, "assets", "demo", D["png"])
if os.path.exists(png):
    c1.image(png, caption="Predicted masks on the max-tumor slice (organ green, tumor red)")
c2.markdown(f"The per-dataset AUSAM model segments **{organ}** and its **tumor** over the full volume. "
            f"Predicted {organ} volume **{_org(pred, organ)['organ_volume_cm3']} cc**, "
            f"tumor **{_org(pred, organ)['tumor_volume_cm3']} cc**.")

# ============================ 2 · Phenotype ============================
st.header("2 · Phenotype")
po = _org(pred, organ)
rows = [["organ volume (cc)", po["organ_volume_cm3"], (_org(gt, organ) or {}).get("organ_volume_cm3", "—")],
        ["tumor volume (cc)", po["tumor_volume_cm3"], (_org(gt, organ) or {}).get("tumor_volume_cm3", "—")],
        ["burden", po["burden_cat"], (_org(gt, organ) or {}).get("burden_cat", "—")],
        ["multiplicity", po["multiplicity"], (_org(gt, organ) or {}).get("multiplicity", "—")],
        ["containment", po["containment"], (_org(gt, organ) or {}).get("containment", "—")],
        ["location", po["anatomic_location"], (_org(gt, organ) or {}).get("anatomic_location", "—")]]
st.table(pd.DataFrame(rows, columns=["phenotype", "predicted", "ground truth (reference)"]))

# ============================ 3 · Validate ============================
st.header("3 · Validate against the train-KG atlas (no labels)")
band = ATLAS.get(organ, {}).get("volume_cm3", {})
lo, hi = band.get("p2.5"), band.get("p97.5")
vol = po["organ_volume_cm3"]
ok = (lo is not None and lo <= vol <= hi)
if ok:
    st.success(f"✅ **Plausible** — predicted {organ} volume {vol} cc is within the cohort band "
               f"[{lo:.0f}, {hi:.0f}] cc (median {band.get('median','?')}). Admitted to the graph.")
else:
    st.error(f"⚠️ **Flagged** — predicted {organ} volume {vol} cc falls outside the plausible band "
             f"[{lo:.0f}, {hi:.0f}] cc — a likely segmentation error, caught with **no ground truth**.")

# ============================ 4 · Grow the KG ============================
st.header("4 · Grow the knowledge graph")
anchor_id = f"{D['case']}·new"
patient = json.loads(json.dumps(pred)); patient["case_id"] = anchor_id     # add as a NEW node
KB_grown = [r for r in KB if r["case_id"] != D["case"]] + [patient]
st.markdown(f"Knowledge base: **{len(KB):,} → {len(KB_grown):,}** patients — this patient is now a node "
            f"grounded to its organ/tumor concepts and is queryable across every dataset.")

# ============================ 5 · Retrieve similar patients ============================
st.header("5 · Retrieve similar patients across the whole knowledge base")
k = st.slider("N similar patients", 3, 15, 8)
method = st.radio("Method", ["OAKG (γ-weighted)", "Masked cosine"], horizontal=True)
corpus, X, M = pr.build_corpus(KB_grown)
mname = "OAKG" if method.startswith("OAKG") else "Masked cosine"
res = pr.rank(anchor_id, mname, corpus, X, M, KB_grown, k=k)
st.dataframe(pd.DataFrame(res), use_container_width=True)
st.caption("γ = joint observability (fraction of shared observed organs); OAKG never imputes a missing organ, so "
           "it won't call two patients similar just because both are missing everything else.")

# local KG view: anchor + retrieved neighbours
G = nx.Graph(); G.add_node("ANCHOR")
for r in res[:min(8, len(res))]:
    lbl = f"{r['patient']}\n({r['dataset']})"
    G.add_node(lbl); G.add_edge("ANCHOR", lbl, w=(r["score"] or 0))
fig, ax = plt.subplots(figsize=(7, 4.5))
posn = nx.spring_layout(G, seed=1)
nx.draw_networkx_nodes(G, posn, nodelist=["ANCHOR"], node_color="#e4572e", node_size=1400, ax=ax)
nx.draw_networkx_nodes(G, posn, nodelist=[n for n in G if n != "ANCHOR"], node_color="#3a86ff", node_size=900, ax=ax)
nx.draw_networkx_edges(G, posn, ax=ax, alpha=0.5)
nx.draw_networkx_labels(G, posn, font_size=7, ax=ax)
ax.set_title(f"{anchor_id} and its {min(8,len(res))} nearest patients ({mname})"); ax.axis("off")
st.pyplot(fig)

# ============================ 6 · Query the whole KB ============================
st.header("6 · Query the entire knowledge base")
q = st.selectbox("Complex query (OAKG paper set)", list(QUERIES))
hits = [r for r in KB_grown if QUERIES[q](r)]
st.markdown(f"**{len(hits):,}** patients match — across "
            f"{', '.join(sorted({r['dataset'] for r in hits}))}.")
show = [{"patient": r["case_id"], "dataset": r["dataset"], "organs": ", ".join(r.get("observed_organs", []))}
        for r in hits[:25]]
st.dataframe(pd.DataFrame(show), use_container_width=True)
