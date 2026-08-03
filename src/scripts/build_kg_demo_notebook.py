#!/usr/bin/env python3
"""Generate the end-to-end SWOG KG pipeline demo notebook (markdown + runnable code)."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
C = []
def md(t): C.append(new_markdown_cell(t))
def code(t): C.append(new_code_cell(t))

md("""# SWOG — Ontology-Grounded Imaging Knowledge Graph
### End-to-end pipeline demo

**Project:** *Ontology Grounded Knowledge Graphs for Explainable Reasoning for GI Cancers* (SWOG Radiation Oncology Committee).

The full SWOG system is a **multimodal** knowledge graph:

- **Text KG** — clinical: patients, tumors, treatments, metastasis, vascular involvement
- **Visual KG** — imaging phenotypes derived from CT segmentation  ← **this notebook builds this half**
- **Phenotype Bridge** — joins the two on the *patient*

This notebook walks the **imaging (Visual) KG** pipeline end to end:

```
schema.owl (ontology, SNOMED/NCIt)          ── the vocabulary
        │
CT masks ─► phenotype records (per patient) ── the instances
        │   kg_build_graph.py
        ▼
imaging_kg.ttl  +  unified_mmkg.json         ── the persisted KG
        │   kg_query.py
        ▼
patient-level queries                        ── explainable reasoning
```

Datasets: **Pancreas (MSD) 281 · LiTS 131 · FLARE23 1,312** — **1,724 per-patient studies**.""")

md("## 0 · Setup")
code("""import sys, os, json, subprocess
from collections import Counter
# portable: walk up from the working dir until we find the repo (kg/schema.owl)
ROOT = os.getcwd()
while ROOT != "/" and not os.path.exists(f"{ROOT}/kg/schema.owl"):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, f"{ROOT}/src/scripts")
import pandas as pd
pd.set_option("display.max_colwidth", 70)
print("repo root:", ROOT)""")

md("""## 1 · Obtain — the ontology (schema)

The KG is **grounded**: every organ / lesion / phenotype is a class in an OWL ontology, cross-referenced
to clinical terminologies (**SNOMED CT**, **NCIt**). This is what makes the graph *explainable* and
interoperable with the clinical Text KG.

> *Grounding provenance:* pancreas, liver, their tumors, and the pancreas sub-sites use **verified** codes
> (`ontology_mappings.json`, checked by direct SNOMED/NCIt lookup); kidney/spleen use **standard but
> not-yet-independently-verified** codes (marked *supplemental* in `kg_build_graph.py`).""")
code("""from rdflib import Graph, RDF, OWL
schema = Graph(); schema.parse(f"{ROOT}/kg/schema.owl")
classes = sorted(str(s).split('/')[-1] for s in schema.subjects(RDF.type, OWL.Class))
rels    = sorted(str(s).split('/')[-1] for s in schema.subjects(RDF.type, OWL.ObjectProperty))
print(f"Ontology: {len(classes)} classes, {len(rels)} relations\\n")
print("Entity classes :", ", ".join(c for c in classes if c[0].isupper())[:200])
print("Relations      :", ", ".join(rels))""")
code("""# ontology grounding: imaging concept -> SNOMED CT / NCIt code
maps = json.load(open(f"{ROOT}/kg/ontology_mappings.json"))["mappings"]
rows = [{"concept": k, **{s: d.get("code") for s, d in v.items()}} for k, v in maps.items()]
pd.DataFrame(rows).fillna("—")""")

md("""## 2 · Obtain — the phenotype instances

Every CT study is run through our SAM3 segmentation models; from the masks we extract a **per-patient
phenotype record**: which organs are observed, tumor presence, **burden** (low/med/high), **multiplicity**
(solitary/multifocal), **containment**, anatomic **sub-site**, and cross-organ extension. These are the
graph's instances.""")
code("""corpus = json.load(open(f"{ROOT}/kg/data/corpus_perpatient.json"))["records"]
print("per-patient studies:", len(corpus), "|", dict(Counter(r["dataset"] for r in corpus)))
ex = next(r for r in corpus if r["dataset"] == "pancreas" and r["organs"]["pancreas"]["has_tumor"])
print("\\nexample patient:", ex["case_id"], "— observed organs:", ex["observed_organs"])
pd.json_normalize(ex["organs"]["pancreas"]).T.rename(columns={0: ex["case_id"] + " · pancreas"})""")

md("""## 3 · Obtain — build the persisted KG

`kg_build_graph.py` materializes the graph: `ImagingCase → Organ → Lesion → Observation`, adds a
**Patient** node per study (the bridge key), grounds concepts to SNOMED/NCIt, and writes two artifacts:

- **`imaging_kg.ttl`** — canonical RDF/Turtle (loads in any triplestore)
- **`unified_mmkg.json`** — nodes+edges (fast to load / app-ingestible)""")
code("""out = subprocess.run([sys.executable, f"{ROOT}/src/scripts/kg_build_graph.py"],
                     capture_output=True, text=True, cwd=ROOT)
print(out.stdout.strip() or out.stderr[-500:])""")

md("## 4 · Inspect the graph")
code("""G = json.load(open(f"{ROOT}/kg/graph/unified_mmkg.json"))
nodes = {n["id"]: n for n in G["nodes"]}
print("Nodes by type :", dict(Counter(n["type"] for n in G["nodes"])))
print("Edges by rel  :", dict(Counter(e["relation"] for e in G["edges"])))""")
md("""### One patient's subgraph
The graph below is the neighborhood of a single pancreatic-cancer patient — case → patient, the depicted
organ, its lesion, the phenotype observations, the anatomic sub-site, and the grounded ontology concepts.
This is the unit the whole KG is built from.""")
code("""import networkx as nx, matplotlib.pyplot as plt
adj = {}
for e in G["edges"]:
    adj.setdefault(e["source"], []).append((e["relation"], e["target"]))
seed = "case_pancreas_001"; keep = {seed}; frontier = [seed]
for _ in range(4):
    nxt = []
    for n in frontier:
        for _, t in adj.get(n, []):
            if t not in keep: keep.add(t); nxt.append(t)
    frontier = nxt
NG = nx.DiGraph()
for e in G["edges"]:
    if e["source"] in keep and e["target"] in keep:
        NG.add_edge(e["source"], e["target"], label=e["relation"])
COLOR = {"ImagingCase":"#4e79a7","Patient":"#f28e2b","Organ":"#59a14f","Lesion":"#e15759",
         "TumorBurden":"#b07aa1","LesionMultiplicity":"#b07aa1","OrganContainment":"#b07aa1",
         "AnatomicSite":"#9c755f","OntologyConcept":"#edc948"}
def short(nid):
    n = nodes.get(nid, {}); lab = n.get("label", nid.split("_")[-1])
    return f"{n.get('type','?')}\\n{lab}"[:26]
pos = nx.spring_layout(NG, seed=3, k=1.1)
plt.figure(figsize=(12, 7))
nx.draw_networkx_nodes(NG, pos, node_size=1700,
    node_color=[COLOR.get(nodes.get(n,{}).get("type"), "#bab0ac") for n in NG])
nx.draw_networkx_edges(NG, pos, edge_color="#888", arrows=True, arrowsize=12, node_size=1700)
nx.draw_networkx_labels(NG, pos, {n: short(n) for n in NG}, font_size=7)
nx.draw_networkx_edge_labels(NG, pos, {(u,v): d["label"] for u,v,d in NG.edges(data=True)}, font_size=6.5)
plt.title(f"KG subgraph — patient pancreas_001  ({NG.number_of_nodes()} nodes, {NG.number_of_edges()} edges)")
plt.axis("off"); plt.tight_layout(); plt.show()""")

md("""## 5 · Validate

Two checks: (a) **ontology grounding** — imaging concepts resolve to SNOMED/NCIt; (b) **phenotype
extraction quality** — extracted (from prediction) vs ground-truth phenotypes.""")
code("""grounded = Counter(e["relation"] for e in G["edges"])["mapped_to_concept"]
concepts = [n for n in G["nodes"] if n["type"] == "OntologyConcept"]
print(f"OntologyConcept nodes: {len(concepts)} | mapped_to_concept edges: {grounded}")
try:
    t2 = json.load(open(f"{ROOT}/kg/data/table2_phenotype_extraction.json"))
    print("\\nphenotype extraction quality (extracted vs GT):")
    print(json.dumps(t2, indent=0)[:400])
except Exception as e:
    print("(table2 phenotype-quality JSON:", e, ")")""")

md("""## 6 · Evaluate — is the reasoning mechanism sound?

The KG's core retrieval mechanism is **observability-aware**: two patients are compared only over the
anatomy they *jointly observe* (unobserved ≠ absent). We validated this against a **coverage-blind**
ablation on the 3-regime corpus. Headline numbers below (precomputed by `kg_controlled_benchmark.py`):
a **zero** gap on the control stratum (as it must be) and a **significant positive** gap where coverage
differs — plus the sharp specificity controls.""")
code("""ev = json.load(open(f"{ROOT}/kg/data/tables_5to11_controlled.json"))
t6, t8 = ev["table6"], ev["table8"]
strat = pd.DataFrame([
    {"stratum": s, "proposed nDCG": t6[s]["proposed"]["nDCG"],
     "coverage-blind": t6[s]["coverage_blind"]["nDCG"], "Δ observability": t6[s]["dObs_nDCG"]}
    for s in ["Within-dataset (control)", "Cross-dataset (target)", "Decomposition (target)"]])
print("Table 6 — retrieval by stratum (proposed vs coverage-blind ablation):")
display(strat)
print(f"\\nTable 8 — coverage controls ({t8.get('n_disjoint_pairs','?'):,} disjoint pairs): "
      f"proposed excludes {t8['incomparable_excluded_pct']['proposed']}% / "
      f"{t8['silence_false_penalty_pct']['proposed']}% false-penalty  |  "
      f"coverage-blind {t8['incomparable_excluded_pct']['coverage_blind']}% / "
      f"{t8['silence_false_penalty_pct']['coverage_blind']}%  -> the mechanism is doing the work.")""")

md("""## 7 · Query — patient-level reasoning

`kg_query.py` runs **complex, multi-hop, patient-level** queries over the graph — the kind of
explainable cohort questions the KG exists to answer.""")
code("""from kg_query import KG, QUERIES
kg = KG()
print(f"KG loaded: {len(kg.patients)} patients, {len(kg.nodes)} nodes")
res = QUERIES(kg)
pd.DataFrame([{"query": k, "count": (len(v) if isinstance(v, list) else v)} for k, v in res.items()])""")

md("""> **Reading the results (data honesty).** Tumor queries resolve across all three datasets:
> **Pancreas** and **LiTS** carry per-patient tumor annotations, and the full **FLARE23** set (1,312
> patients) adds tumor on **liver, kidney, and pancreas** for **608** patients, plus 13-organ coverage
> for observability. So patient-level *tumor* reasoning now spans Pancreas + LiTS + FLARE23, and FLARE23
> also provides the multi-organ (up to 13) coverage that drives the observability-aware retrieval.""")

md("""## 8 · The SWOG use-case — Q1: *Whipple surgery, lymph-node recurrence*

The flagship SWOG query is **multimodal**: *Whipple surgery* and *lymph-node recurrence* live in the
**Text KG**; the **imaging** half is *"patients with a pancreatic-head tumor"* (Whipple = pancreatic-head
resection). Our KG answers that half and exposes each match by **`patient_id`** — the join key the
Phenotype Bridge uses to combine with the clinical KG.""")
code("""whipple_cohort = kg.patients_where(
    lambda c: any(kg.at_site(l, "head") for _, l in kg.organ_lesions(c, "pancreas")))
print(f"Whipple-relevant imaging cohort (pancreatic-head tumor): {len(whipple_cohort)} patients\\n")
rows = []
for p in whipple_cohort[:8]:
    cid = kg.prop(p, "patient_id"); o, l = next(kg.organ_lesions(f"case_{cid}", "pancreas"))
    rows.append({"patient_id": cid, "burden": kg.obs_value(l, "TumorBurden"),
                 "containment": kg.obs_value(l, "OrganContainment"),
                 "tumor_cm3": kg.prop(l, "volume_cm3"),
                 "→ bridge to Text KG": "Whipple? recurrence?"})
pd.DataFrame(rows)""")

md("""## Summary

| Stage | Artifact | Status |
|---|---|---|
| **Obtain** | `schema.owl` + phenotype instances → `imaging_kg.ttl` / `unified_mmkg.json` | ✅ persisted KG |
| **Validate** | ontology grounding (SNOMED/NCIt), phenotype quality | ✅ |
| **Evaluate** | observability-aware retrieval / integration (`kg_*` scripts) | ✅ |
| **Query** | patient-level complex queries (`kg_query.py`) | ✅ |

**This half is done.** The imaging KG is patient-keyed and ready to bridge into the clinical **Text KG**
(the other half of the multimodal system) to answer end-to-end queries like Q1.

**Next:** integrate the Text KG on `patient_id` so *Whipple + recurrence + imaging phenotype* runs as one query.""")

nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
path = "/home/ud3d4/Desktop/SWOG/src/notebooks/SWOG_KG_Pipeline_Demo.ipynb"
import os
os.makedirs(os.path.dirname(path), exist_ok=True)
nbf.write(nb, path)
print("wrote", path, f"({len(C)} cells)")
