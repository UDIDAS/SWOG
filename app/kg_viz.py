#!/usr/bin/env python3
"""
KG visualizations for the OAKG-retrieved patients.

Streamlit-free so it is reusable/testable:
  patient_graph_html(rec, mappings)  -> interactive vis.js graph (drag/zoom/hover) for ONE patient
  cohort_strip_figure(records, ...)  -> queried-phenotype value per patient, by dataset (Plotly)
  cohort_bar_figure(records)         -> simple "retrieved patients per dataset" bar (Plotly)
  patient_bars_figure(rec, norgan)   -> organ vs tumor volume for one patient (Plotly)

The graph inlines the vendored vis-network JS (app/assets/vis-network.min.js) so it is fully
self-contained and offline — we deliberately do NOT import pyvis (its IPython->sqlite3 import
chain is broken in this env's library load order).
"""
import json
import os
from collections import Counter

import pandas as pd
import plotly.express as px

_VIS_JS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets",
                            "vis-network.min.js")
with open(_VIS_JS_PATH) as _f:
    _VIS_JS = _f.read()

TYPE_COLOR = {
    "Patient": "#d62728", "ImagingCase": "#1f77b4", "Organ": "#2ca02c",
    "Lesion": "#ff7f0e", "Observation": "#9467bd", "AnatomicSite": "#17becf",
    "Concept": "#7f7f7f", "Dataset": "#8c564b",
}
ORGAN_MAP = {"pancreas": "Organ::Pancreas", "liver": "Organ::Liver", "spleen": "Organ::Spleen",
             "kidney": "Organ::Kidney", "right_kidney": "Organ::Kidney", "left_kidney": "Organ::Kidney"}
TUMOR_MAP = {"pancreas": "Lesion::Pancreatic tumor", "liver": "Lesion::Liver tumor",
             "kidney": "Lesion::Kidney tumor", "right_kidney": "Lesion::Kidney tumor",
             "left_kidney": "Lesion::Kidney tumor"}
SITE_MAP = {"head": "AnatomicSite::Head of pancreas", "body": "AnatomicSite::Body of pancreas",
            "tail": "AnatomicSite::Tail of pancreas"}
# categorical phenotypes as DIRECT predicate triples: (lesion) --tumorBurden--> "high"
OBS_PRED = {"burden_cat": "tumorBurden", "multiplicity": "lesionMultiplicity",
            "containment": "organContainment"}

_OPTIONS = {
    "interaction": {"hover": True, "tooltipDelay": 80, "navigationButtons": True,
                    "keyboard": True, "multiselect": True},
    "physics": {"solver": "barnesHut",
                "barnesHut": {"gravitationalConstant": -14000, "springLength": 130,
                              "springConstant": 0.035, "damping": 0.45},
                "minVelocity": 0.5, "stabilization": {"iterations": 160}},
    "nodes": {"font": {"size": 15, "face": "arial"}, "borderWidth": 1},
    "edges": {"font": {"size": 11, "align": "top", "color": "#555"}, "smooth": False,
              "color": {"color": "#bbbbbb", "highlight": "#555"},
              "arrows": {"to": {"enabled": True, "scaleFactor": 0.5}}},
}


def _concept(mappings, mapkey):
    d = mappings.get(mapkey, {})
    for sysname in ("SNOMED CT", "NCIt"):
        c = d.get(sysname)
        if c and c.get("code"):
            return f"{sysname} {c['code']}", c.get("display")
    return None, None


def patient_graph_html(rec, mappings, height=560):
    """Self-contained interactive vis.js graph of one patient's KG subgraph (drag/zoom/hover)."""
    nodes, edges, seen = [], [], set()

    def add(nid, label, ntype, title, size=18, shape="dot"):
        if nid not in seen:
            seen.add(nid)
            nodes.append({"id": nid, "label": label, "title": title, "shape": shape,
                          "size": size, "color": TYPE_COLOR[ntype], "group": ntype})

    def link(a, b, rel):
        edges.append({"from": a, "to": b, "label": rel})

    cid = rec["case_id"]
    add(f"P:{cid}", cid, "Patient", f"Patient — id {cid}, dataset {rec['dataset']}", 30, "star")
    add(f"C:{cid}", "ImagingCase", "ImagingCase",
        f"ImagingCase — CT; observed organs: {', '.join(rec['observed_organs'])}", 22, "square")
    link(f"P:{cid}", f"C:{cid}", "of_patient")

    for o in rec["observed_organs"]:
        od = rec["organs"][o]
        add(f"O:{o}", o.replace("_", " "), "Organ",
            f"Organ {o.replace('_', ' ')} — volume {od.get('organ_volume_cm3')} cm³", 22)
        link(f"C:{cid}", f"O:{o}", "depicts_organ")
        code, disp = _concept(mappings, ORGAN_MAP.get(o, ""))
        if code:
            add(f"KO:{o}", code, "Concept", f"{code} — {disp}", 14, "box")
            link(f"O:{o}", f"KO:{o}", "mapped_to_concept")
        if od.get("has_tumor"):
            add(f"L:{o}", f"{o.replace('_', ' ')} tumor", "Lesion",
                f"Lesion — volume {od.get('tumor_volume_cm3')} cm³, voxels {od.get('tumor_voxels')}",
                22, "triangle")
            link(f"O:{o}", f"L:{o}", "has_lesion")
            code, disp = _concept(mappings, TUMOR_MAP.get(o, ""))
            if code:
                add(f"KL:{o}", code, "Concept", f"{code} — {disp}", 14, "box")
                link(f"L:{o}", f"KL:{o}", "mapped_to_concept")
            for field, pred in OBS_PRED.items():
                v = od.get(field)
                if v in (None, "none", "na", "unknown"):
                    continue
                add(f"OB:{o}:{field}", str(v), "Observation",
                    f"{pred} = {v} (ground truth)", 16, "ellipse")
                link(f"L:{o}", f"OB:{o}:{field}", pred)          # direct triple: lesion -pred-> value
            loc = od.get("anatomic_location")
            if loc in SITE_MAP:
                add(f"S:{o}", f"{loc} of {o.replace('_', ' ')}", "AnatomicSite",
                    f"AnatomicSite — {loc} of {o.replace('_', ' ')}", 16, "diamond")
                link(f"L:{o}", f"S:{o}", "at_site")
                code, disp = _concept(mappings, SITE_MAP[loc])
                if code:
                    add(f"KS:{o}", code, "Concept", f"{code} — {disp}", 14, "box")
                    link(f"S:{o}", f"KS:{o}", "mapped_to_concept")

    return _wrap(nodes, edges, height)


def _wrap(nodes, edges, height):
    """Wrap nodes/edges into a self-contained interactive vis.js HTML document."""
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>#net{{width:100%;height:{height}px;border:1px solid #eee;border-radius:6px}}
body{{margin:0;font-family:arial,sans-serif}}</style>
<script type="text/javascript">{_VIS_JS}</script></head><body>
<div id="net"></div>
<script type="text/javascript">
  var nodes = new vis.DataSet({json.dumps(nodes)});
  var edges = new vis.DataSet({json.dumps(edges)});
  var network = new vis.Network(document.getElementById('net'),
                                {{nodes: nodes, edges: edges}}, {json.dumps(_OPTIONS)});
</script></body></html>"""


def merged_kg_html(records_list, mappings, height=560):
    """One KG over MANY patients — they connect through SHARED Dataset + OntologyConcept nodes
    (the shared schema). Per-patient nodes are id-scoped by case; Dataset/Concept nodes are global."""
    nodes, edges, seen = [], [], set()

    def add(nid, label, ntype, title, size=16, shape="dot"):
        if nid not in seen:
            seen.add(nid)
            nodes.append({"id": nid, "label": label, "title": title, "shape": shape,
                          "size": size, "color": TYPE_COLOR[ntype], "group": ntype})

    def link(a, b, rel):
        edges.append({"from": a, "to": b, "label": rel})

    for rec in records_list:
        cid, ds = rec["case_id"], rec["dataset"]
        add(f"DS:{ds}", ds, "Dataset", f"Dataset: {ds}", 26, "database")          # SHARED
        add(f"P:{cid}", cid, "Patient", f"Patient {cid} ({ds})", 18, "star")
        add(f"C:{cid}", "case", "ImagingCase", f"ImagingCase — {cid}", 12, "square")
        link(f"P:{cid}", f"C:{cid}", "of_patient")
        link(f"C:{cid}", f"DS:{ds}", "from_dataset")
        for o in rec["observed_organs"]:
            od = rec["organs"][o]
            add(f"O:{cid}:{o}", o.replace("_", " "), "Organ",
                f"{o.replace('_', ' ')} — {od.get('organ_volume_cm3')} cm³ ({cid})", 12)
            link(f"C:{cid}", f"O:{cid}:{o}", "depicts_organ")
            code, disp = _concept(mappings, ORGAN_MAP.get(o, ""))
            if code:
                add(f"K:{code}", code, "Concept", f"{code} — {disp}", 16, "box")   # SHARED
                link(f"O:{cid}:{o}", f"K:{code}", "mapped_to_concept")
            if od.get("has_tumor"):
                add(f"L:{cid}:{o}", f"{o.replace('_', ' ')} tumor", "Lesion",
                    f"tumor — {od.get('tumor_volume_cm3')} cm³, burden {od.get('burden_cat')} ({cid})",
                    12, "triangle")
                link(f"O:{cid}:{o}", f"L:{cid}:{o}", "has_lesion")
                code, disp = _concept(mappings, TUMOR_MAP.get(o, ""))
                if code:
                    add(f"K:{code}", code, "Concept", f"{code} — {disp}", 16, "box")  # SHARED
                    link(f"L:{cid}:{o}", f"K:{code}", "mapped_to_concept")
    return _wrap(nodes, edges, height)


def cohort_strip_figure(records, nlabel, norgan, nfield):
    df = pd.DataFrame([{"patient": r["case_id"], "dataset": r["dataset"],
                        nlabel: r["organs"].get(norgan, {}).get(nfield)}
                       for r in records if norgan in r["organs"]])
    fig = px.strip(df, x="dataset", y=nlabel, color="dataset", hover_data=["patient"],
                   title=f"{nlabel} of retrieved patients, by dataset")
    fig.update_traces(jitter=1.0, marker=dict(size=8, opacity=0.65))
    fig.update_layout(showlegend=False, height=360, margin=dict(l=10, r=10, t=44, b=10))
    return fig


def cohort_bar_figure(records):
    c = Counter(r["dataset"] for r in records)
    df = pd.DataFrame({"dataset": list(c), "patients": list(c.values())}).sort_values("patients")
    fig = px.bar(df, x="patients", y="dataset", color="dataset", orientation="h", text="patients",
                 title="How many retrieved patients come from each dataset")
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, height=360, margin=dict(l=10, r=10, t=44, b=10),
                      xaxis_title="patients returned", yaxis_title="")
    return fig


def patient_bars_figure(rec, norgan):
    od = rec["organs"].get(norgan, {})
    df = pd.DataFrame({"measure": ["organ volume", "tumor volume"],
                       "cm³": [od.get("organ_volume_cm3") or 0, od.get("tumor_volume_cm3") or 0]})
    fig = px.bar(df, x="measure", y="cm³", color="measure", text="cm³",
                 title=f"{rec['case_id']} — {norgan.replace('_', ' ')} volumes")
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, height=320, margin=dict(l=10, r=10, t=44, b=10))
    return fig
