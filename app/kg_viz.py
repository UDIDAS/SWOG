#!/usr/bin/env python3
"""
Interactive KG visualizations for the OAKG-retrieved patients (Plotly).

Streamlit-free so it is reusable/testable:
  patient_kg_figure(rec, mappings)  -> node-link KG subgraph for ONE patient (layered tree)
  cohort_strip_figure(records, ...) -> queried-phenotype value per patient, by dataset
  cohort_sunburst_figure(records..) -> dataset -> organ -> tumor -> burden composition
  patient_bars_figure(rec, ...)     -> organ vs tumor volume for one patient
"""
import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

TYPE_COLOR = {
    "Patient": "#d62728", "ImagingCase": "#1f77b4", "Organ": "#2ca02c",
    "Lesion": "#ff7f0e", "Observation": "#9467bd", "AnatomicSite": "#bcbd22",
    "Concept": "#7f7f7f",
}
ORGAN_MAP = {"pancreas": "Organ::Pancreas", "liver": "Organ::Liver"}
TUMOR_MAP = {"pancreas": "Lesion::Pancreatic tumor", "liver": "Lesion::Liver tumor"}
SITE_MAP = {"head": "AnatomicSite::Head of pancreas", "body": "AnatomicSite::Body of pancreas",
            "tail": "AnatomicSite::Tail of pancreas"}
OBS_CLS = {"burden_cat": "TumorBurden", "multiplicity": "LesionMultiplicity",
           "containment": "OrganContainment"}


def _concept(mappings, mapkey):
    d = mappings.get(mapkey, {})
    for sysname in ("SNOMED CT", "NCIt"):
        c = d.get(sysname)
        if c and c.get("code"):
            return f"{sysname} {c['code']}", c.get("display")
    return None, None


def patient_kg_figure(rec, mappings):
    """Layered node-link graph of one patient's KG subgraph."""
    G = nx.DiGraph()
    cid = rec["case_id"]

    def add(nid, layer, ntype, label, hover):
        G.add_node(nid, layer=layer, ntype=ntype, label=label, hover=hover)

    pn = f"P:{cid}"
    add(pn, 0, "Patient", cid, f"<b>Patient</b><br>id: {cid}<br>dataset: {rec['dataset']}")
    cn = f"C:{cid}"
    add(cn, 1, "ImagingCase", "ImagingCase",
        f"<b>ImagingCase</b><br>modality: CT<br>dataset: {rec['dataset']}"
        f"<br>observed organs: {', '.join(rec['observed_organs'])}")
    G.add_edge(pn, cn, rel="of_patient")

    for o in rec["observed_organs"]:
        od = rec["organs"][o]
        on = f"O:{o}"
        add(on, 2, "Organ", o.replace("_", " "),
            f"<b>Organ: {o.replace('_', ' ')}</b><br>volume: {od.get('organ_volume_cm3')} cm³")
        G.add_edge(cn, on, rel="depicts_organ")
        code, disp = _concept(mappings, ORGAN_MAP.get(o, ""))
        if code:
            kn = f"KO:{o}"
            add(kn, 3, "Concept", code, f"<b>OntologyConcept</b><br>{code}<br>{disp}")
            G.add_edge(on, kn, rel="mapped_to_concept")
        if od.get("has_tumor"):
            ln = f"L:{o}"
            add(ln, 3, "Lesion", f"{o.replace('_', ' ')} tumor",
                f"<b>Lesion (tumor)</b><br>volume: {od.get('tumor_volume_cm3')} cm³"
                f"<br>voxels: {od.get('tumor_voxels')}")
            G.add_edge(on, ln, rel="has_lesion")
            code, disp = _concept(mappings, TUMOR_MAP.get(o, ""))
            if code:
                kn = f"KL:{o}"
                add(kn, 4, "Concept", code, f"<b>OntologyConcept</b><br>{code}<br>{disp}")
                G.add_edge(ln, kn, rel="mapped_to_concept")
            for field, cls in OBS_CLS.items():
                v = od.get(field)
                if v in (None, "none", "na", "unknown"):
                    continue
                nn = f"OB:{o}:{field}"
                add(nn, 4, "Observation", f"{cls}={v}",
                    f"<b>{cls}</b><br>value: {v}<br>source: ground_truth")
                G.add_edge(ln, nn, rel="has_observation")
            loc = od.get("anatomic_location")
            if loc in SITE_MAP:
                sn = f"S:{o}"
                add(sn, 4, "AnatomicSite", f"{loc} of {o.replace('_', ' ')}",
                    f"<b>AnatomicSite</b><br>{loc} of {o.replace('_', ' ')}")
                G.add_edge(ln, sn, rel="at_site")
                code, disp = _concept(mappings, SITE_MAP[loc])
                if code:
                    kn = f"KS:{o}"
                    add(kn, 4, "Concept", code, f"<b>OntologyConcept</b><br>{code}<br>{disp}")
                    G.add_edge(sn, kn, rel="mapped_to_concept")

    pos = nx.multipartite_layout(G, subset_key="layer")
    fig = go.Figure()

    # edges + relation labels
    ex, ey, lx, ly, lt = [], [], [], [], []
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        ex += [x0, x1, None]; ey += [y0, y1, None]
        lx.append((x0 + x1) / 2); ly.append((y0 + y1) / 2); lt.append(d["rel"])
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#cccccc", width=1),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=lx, y=ly, mode="text", text=lt, textfont=dict(size=8, color="#999"),
                             hoverinfo="skip", showlegend=False))

    # nodes, one trace per type (legend)
    for ntype, color in TYPE_COLOR.items():
        nodes = [n for n, a in G.nodes(data=True) if a["ntype"] == ntype]
        if not nodes:
            continue
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes],
            mode="markers+text", name=ntype,
            text=[G.nodes[n]["label"] for n in nodes], textposition="top center",
            textfont=dict(size=9),
            hovertext=[G.nodes[n]["hover"] for n in nodes], hoverinfo="text",
            marker=dict(size=[22 if ntype == "Patient" else 16 for _ in nodes],
                        color=color, line=dict(width=1, color="white"))))

    fig.update_layout(title=f"KG subgraph — patient {cid} ({rec['dataset']})",
                      height=460, margin=dict(l=10, r=10, t=44, b=10),
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
                      plot_bgcolor="white")
    return fig


def cohort_strip_figure(records, nlabel, norgan, nfield):
    df = pd.DataFrame([{"patient": r["case_id"], "dataset": r["dataset"],
                        nlabel: r["organs"].get(norgan, {}).get(nfield)}
                       for r in records if norgan in r["organs"]])
    fig = px.strip(df, x="dataset", y=nlabel, color="dataset", hover_data=["patient"],
                   title=f"{nlabel} of retrieved patients, by dataset")
    fig.update_traces(jitter=1.0, marker=dict(size=8, opacity=0.65))
    fig.update_layout(showlegend=False, height=380, margin=dict(l=10, r=10, t=44, b=10))
    return fig


def cohort_sunburst_figure(records, norgan):
    rows = []
    for r in records:
        od = r["organs"].get(norgan, {})
        burden = od.get("burden_cat") or "n/a"
        rows.append({"dataset": r["dataset"], "organ": norgan.replace("_", " "),
                     "tumor": "tumor" if od.get("has_tumor") else "no tumor",
                     "burden": "n/a" if burden in ("none", "na", "unknown") else burden, "n": 1})
    fig = px.sunburst(pd.DataFrame(rows), path=["dataset", "organ", "tumor", "burden"], values="n",
                      title="Composition of the retrieved cohort")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=44, b=10))
    return fig


def patient_bars_figure(rec, norgan):
    od = rec["organs"].get(norgan, {})
    df = pd.DataFrame({
        "measure": ["organ volume", "tumor volume"],
        "cm³": [od.get("organ_volume_cm3") or 0, od.get("tumor_volume_cm3") or 0]})
    fig = px.bar(df, x="measure", y="cm³", color="measure", text="cm³",
                 title=f"{rec['case_id']} — {norgan.replace('_', ' ')} volumes")
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, height=320, margin=dict(l=10, r=10, t=44, b=10))
    return fig
