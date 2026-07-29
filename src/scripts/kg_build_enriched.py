#!/usr/bin/env python3
"""
kg_build_enriched.py — Pancreas + LiTS KG carrying BOTH ground-truth and PREDICTED metrics.

Source of the rich metrics = the hand-off manifests (per_label_stats_gt / per_label_stats_pred /
per_case_metrics), joined with the categorical GT phenotypes (burden / multiplicity / containment /
sub-site) from case_phenotypes.json. Every Organ and Lesion node gets:
  gt_volume_cm3 / pred_volume_cm3, gt_max_diameter_mm / pred_max_diameter_mm,
  gt_voxels / pred_voxels, gt_centroid_mm / pred_centroid_mm, dice_score (pred vs GT).
Lesions also get lesion_count and the categorical observations, plus SNOMED/NCIt grounding.

FLARE is intentionally excluded (per request). Out: kg/graph/imaging_kg_pancreas_lits.ttl (+ .json).
"""
import json, os
from collections import Counter
from rdflib import Graph, Namespace, Literal, RDF, XSD, OWL, RDFS

KG = "/home/ud3d4/Desktop/SWOG/kg"
OUT = f"{KG}/graph"
HANDOFF = "/scratch/ud3d4/acm_data/handoff_restore/ssl_handoff_ours"
MMKG = Namespace("http://example.org/mmkg/schema/")
INST = Namespace("http://example.org/mmkg/instance/")

# dataset -> (organ label id, organ name, organ class, organ mapkey, tumor class, tumor mapkey)
DS = {
    "pancreas": dict(organ="pancreas", organ_cls="Pancreas", organ_mk="Organ::Pancreas",
                     tumor_cls="PancreaticTumor", tumor_mk="Lesion::Pancreatic tumor"),
    "lits": dict(organ="liver", organ_cls="Liver", organ_mk="Organ::Liver",
                 tumor_cls="LiverTumor", tumor_mk="Lesion::Liver tumor"),
}
SITE_MK = {"PancreaticHead": "AnatomicSite::Head of pancreas", "head": "AnatomicSite::Head of pancreas",
           "PancreaticBody": "AnatomicSite::Body of pancreas", "body": "AnatomicSite::Body of pancreas",
           "PancreaticTail": "AnatomicSite::Tail of pancreas", "tail": "AnatomicSite::Tail of pancreas"}
SITE_CLS = {"head": "PancreaticHead", "body": "PancreaticBody", "tail": "PancreaticTail"}
# categorical phenotypes as DIRECT predicate triples:  lesion --<pred>--> "value"
OBS_PRED = {"burden_cat": "tumorBurden", "multiplicity": "lesionMultiplicity",
            "containment": "organContainment"}


def uri(*p):
    return INST["_".join(str(x).replace(" ", "").replace(":", "").replace("/", "") for x in p)]


def build(sel, tag, title):
    """Build one enriched KG over datasets in `sel` -> imaging_kg{tag}.ttl (+ unified json)."""
    os.makedirs(OUT, exist_ok=True)
    g = Graph(); g.parse(f"{KG}/schema.owl"); g.bind("mmkg", MMKG); g.bind("inst", INST)
    for pred in OBS_PRED.values():                     # declare the categorical predicates
        g.add((MMKG[pred], RDF.type, OWL.DatatypeProperty))
        g.add((MMKG[pred], RDFS.domain, MMKG.Lesion))
    nodes, edges = {}, []

    def node(nid, ntype, label=None, **props):
        n = nodes.setdefault(str(nid), {"id": str(nid).split("/")[-1], "type": ntype, "properties": {}})
        if label: n["label"] = label
        n["properties"].update({k: v for k, v in props.items() if v is not None})
        return nid
    def edge(s, rel, o):
        g.add((s, MMKG[rel], o)); edges.append({"source": str(s).split("/")[-1], "relation": rel, "target": str(o).split("/")[-1]})
    def typ(u, c): g.add((u, RDF.type, MMKG[c]))
    def lit(u, p, v, dt=None):
        if v is not None: g.add((u, MMKG[p], Literal(v, datatype=dt) if dt else Literal(v)))

    # ontology concepts
    maps = json.load(open(f"{KG}/ontology_mappings.json"))["mappings"]
    cu = {}
    for mk, systems in maps.items():
        first = None
        for s, d in systems.items():
            if not d.get("code"): continue
            u = uri("concept", s, d["code"]); node(u, "OntologyConcept", d.get("display"), system=s, code=d["code"])
            typ(u, "OntologyConcept"); lit(u, "system", s); lit(u, "code", d["code"]); lit(u, "display", d.get("display"))
            first = first or u
        if first: cu[mk] = first

    # GT categoricals join (burden/multiplicity/containment/location) by case_id -> organ
    cats = {}
    for r in json.load(open(f"{KG}/data/case_phenotypes.json"))["records"]:
        for o, od in r["organs"].items():
            cats[(r["case_id"], o)] = od

    def metrics(u, pred, gt, dice, prefix_lesion=False):
        nd = nodes[str(u)]["properties"]                 # keep the JSON export as rich as the TTL
        for src, st in [("gt", gt), ("pred", pred)]:
            if not st: continue
            for key, dt in [("volume_cm3", XSD.float), ("max_diameter_mm", XSD.float), ("voxels", XSD.integer)]:
                v = st.get(key)
                if v is not None:
                    lit(u, f"{src}_{key}", v, dt); nd[f"{src}_{key}"] = v
            if st.get("centroid_mm"):
                lit(u, f"{src}_centroid_mm", str(st["centroid_mm"])); nd[f"{src}_centroid_mm"] = st["centroid_mm"]
        if dice is not None:
            lit(u, "dice_score", dice, XSD.float); nd["dice_score"] = dice

    n_cases = 0
    for ds, info in DS.items():
        if ds not in sel: continue
        d = json.load(open(f"{HANDOFF}/ssl_handoff_{ds}.json"))
        for cid, c in d["cases"].items():
            n_cases += 1
            cuu = uri("case", cid); node(cuu, "ImagingCase", cid, dataset=ds, modality="CT")
            typ(cuu, "ImagingCase"); lit(cuu, "case_id", cid); lit(cuu, "dataset", ds)
            pu = uri("patient", cid); node(pu, "Patient", cid, patient_id=cid, dataset=ds); typ(pu, "Patient")
            lit(pu, "patient_id", cid); edge(cuu, "of_patient", pu)
            du = uri("dataset", ds); node(du, "Dataset", ds); typ(du, "Dataset"); edge(cuu, "from_dataset", du)

            sg, sp = c.get("per_label_stats_gt", {}), c.get("per_label_stats_pred", {})
            m = c.get("per_case_metrics", {})
            # organ (label 1)
            ou = uri("organ", cid, info["organ"])
            node(ou, "Organ", info["organ"])
            typ(ou, "Organ"); typ(ou, info["organ_cls"]); edge(cuu, "depicts_organ", ou)
            metrics(ou, sp.get("1"), sg.get("1"), m.get("label_1", {}).get("dice"))
            if info["organ_mk"] in cu: edge(ou, "mapped_to_concept", cu[info["organ_mk"]])

            # lesion (label 2) — GT tumor present
            if sg.get("2"):
                lu = uri("lesion", cid, "tumor")
                node(lu, "Lesion", f"{info['organ']} tumor", is_tumor=True)
                typ(lu, "Lesion"); typ(lu, info["tumor_cls"]); lit(lu, "is_tumor", True, XSD.boolean)
                edge(ou, "has_lesion", lu); edge(lu, "located_in", ou)
                metrics(lu, sp.get("2"), sg.get("2"), m.get("label_2", {}).get("dice"), True)
                if info["tumor_mk"] in cu: edge(lu, "mapped_to_concept", cu[info["tumor_mk"]])
                dp = c.get(f"derived_phenotype_{ds}", {})
                lit(lu, "lesion_count", dp.get("tumor_lesion_count"), XSD.integer)
                # categorical phenotypes as DIRECT triples: (lesion, tumorBurden, "low") etc. (GT-derived)
                cat = cats.get((cid, info["organ"]), {})
                for field, pred in OBS_PRED.items():
                    v = cat.get(field)
                    if v in (None, "none", "na", "unknown"): continue
                    lit(lu, pred, v); nodes[str(lu)]["properties"][pred] = v
                loc = dp.get("anatomic_location") or cat.get("anatomic_location")
                if loc in SITE_MK:
                    su = uri("site", loc); node(su, "AnatomicSite", loc); typ(su, SITE_CLS.get(str(loc).lower().replace("pancreatic","").strip(), "AnatomicSite"))
                    edge(lu, "at_site", su)
                    if SITE_MK[loc] in cu: edge(su, "mapped_to_concept", cu[SITE_MK[loc]])

    g.serialize(f"{OUT}/imaging_kg{tag}.ttl", format="turtle")
    json.dump({"nodes": list(nodes.values()), "edges": edges, "schema": "kg/schema.owl",
               "datasets": sel, "note": "GT + predicted metrics per organ/lesion"},
              open(f"{OUT}/unified_mmkg{tag}.json", "w"))
    print(f"{title}: {n_cases} cases, {len(g)} triples -> {OUT}/imaging_kg{tag}.ttl")
    print("  Nodes:", dict(Counter(n["type"] for n in nodes.values())),
          "| Edges:", dict(Counter(e["relation"] for e in edges)))


if __name__ == "__main__":
    build(["pancreas"], "_pancreas", "Enriched Pancreas KG")
    build(["lits"], "_lits", "Enriched LiTS KG")
    build(["pancreas", "lits"], "_pancreas_lits", "Enriched Pancreas+LiTS KG")
