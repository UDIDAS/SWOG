#!/usr/bin/env python3
"""FLARE23 GT-only ENRICHED KG — same schema as the Pancreas/LiTS handoff (kg_build_enriched.py):
direct categorical triples on the Lesion (mmkg:tumorBurden / mmkg:lesionMultiplicity), gt_ metrics,
Organ/Lesion nodes, SNOMED/NCIt grounding. FLARE23 is GT-derived and prediction-free, so only gt_*.

in:  kg/data/corpus_flare_train.json   out: kg/graph/imaging_kg_flare23.ttl (+ unified_mmkg_flare23.json)
"""
import json
import os
from collections import Counter

from rdflib import Graph, Literal, Namespace, OWL, RDF, RDFS, XSD

KG = "/home/ud3d4/Desktop/SWOG/kg"
OUT = f"{KG}/graph"
CORPUS = f"{KG}/data/corpus_flare_train.json"
MMKG = Namespace("http://example.org/mmkg/schema/")
INST = Namespace("http://example.org/mmkg/instance/")

# organ -> (schema class, SNOMED CT, NCIt) — all 13 FLARE organs (abdominal CT scope)
ORGAN_CONCEPT = {
    "liver": ("Liver", "10200004", "C12392"),
    "right_kidney": ("Kidney", "9846003", "C12420"),
    "spleen": ("Spleen", "78961009", "C12432"),
    "pancreas": ("Pancreas", "15776009", "C12393"),
    "aorta": ("Aorta", "15825003", "C12683"),
    "inferior_vena_cava": ("InferiorVenaCava", "64131007", "C12685"),
    "right_adrenal": ("AdrenalGland", "23451007", "C12666"),
    "left_adrenal": ("AdrenalGland", "23451007", "C12666"),
    "gallbladder": ("Gallbladder", "28231008", "C12377"),
    "esophagus": ("Esophagus", "32849002", "C12389"),
    "stomach": ("Stomach", "69695003", "C12391"),
    "duodenum": ("Duodenum", "38848004", "C12263"),
    "left_kidney": ("Kidney", "18639004", "C12421"),
}
TUMOR_CONCEPT = {                       # organ-specific tumor grounding where a code exists
    "liver": ("LiverTumor", "93870000", "C3099"),
    "pancreas": ("PancreaticTumor", "372003004", "C3305"),
}
OBS_PRED = {"burden_cat": "tumorBurden", "multiplicity": "lesionMultiplicity"}


def uri(*p):
    return INST["_".join(str(x).replace(" ", "").replace(":", "").replace("/", "") for x in p)]


def main():
    os.makedirs(OUT, exist_ok=True)
    g = Graph()
    g.parse(f"{KG}/schema.owl")
    g.bind("mmkg", MMKG)
    g.bind("inst", INST)
    for pred in OBS_PRED.values():
        g.add((MMKG[pred], RDF.type, OWL.DatatypeProperty))
        g.add((MMKG[pred], RDFS.domain, MMKG.Lesion))
    nodes, edges = {}, []

    def node(nid, ntype, label=None, **props):
        n = nodes.setdefault(str(nid), {"id": str(nid).split("/")[-1], "type": ntype, "properties": {}})
        if label:
            n["label"] = label
        n["properties"].update({k: v for k, v in props.items() if v is not None})
        return nid

    def edge(s, rel, o):
        g.add((s, MMKG[rel], o))
        edges.append({"source": str(s).split("/")[-1], "relation": rel, "target": str(o).split("/")[-1]})

    def typ(u, c):
        g.add((u, RDF.type, MMKG[c]))

    def lit(u, p, v, dt=None):
        if v is not None:
            g.add((u, MMKG[p], Literal(v, datatype=dt) if dt else Literal(v)))

    concept = {}

    def add_concept(key, label, codes):
        first = None
        for sysname, code in codes:
            if not code:
                continue
            u = uri("concept", sysname, code)
            node(u, "OntologyConcept", label, system=sysname, code=code)
            typ(u, "OntologyConcept")
            lit(u, "system", sysname)
            lit(u, "code", code)
            lit(u, "display", label)
            first = first or u
        if first:
            concept[key] = first

    for o, (cls, sn, nc) in ORGAN_CONCEPT.items():
        add_concept(("organ", o), cls, [("SNOMED CT", sn), ("NCIt", nc)])
    for o, (cls, sn, nc) in TUMOR_CONCEPT.items():
        add_concept(("tumor", o), cls, [("SNOMED CT", sn), ("NCIt", nc)])

    recs = json.load(open(CORPUS))["records"]
    n_cases = n_les = 0
    for r in recs:
        cid, ds = r["case_id"], "flare"
        cuu = uri("case", cid)
        node(cuu, "ImagingCase", cid, dataset=ds, modality="CT")
        typ(cuu, "ImagingCase")
        lit(cuu, "case_id", cid)
        lit(cuu, "dataset", ds)
        pu = uri("patient", cid)
        node(pu, "Patient", cid, patient_id=cid, dataset=ds)
        typ(pu, "Patient")
        lit(pu, "patient_id", cid)
        edge(cuu, "of_patient", pu)
        du = uri("dataset", ds)
        node(du, "Dataset", ds)
        typ(du, "Dataset")
        edge(cuu, "from_dataset", du)
        n_cases += 1
        for o, od in r["organs"].items():
            ou = uri("organ", cid, o)
            node(ou, "Organ", o.replace("_", " "))
            typ(ou, "Organ")
            oc = ORGAN_CONCEPT.get(o)
            if oc:
                typ(ou, oc[0])
            edge(cuu, "depicts_organ", ou)
            for src, gk, dt in [("organ_volume_cm3", "gt_volume_cm3", XSD.float),
                                ("organ_voxels", "gt_voxels", XSD.integer),
                                ("organ_max_diameter_mm", "gt_max_diameter_mm", XSD.float)]:
                val = od.get(src)
                if val is not None:
                    lit(ou, gk, val, dt)
                    nodes[str(ou)]["properties"][gk] = val
            cen = od.get("organ_centroid_mm")
            if cen:
                lit(ou, "gt_centroid_mm", str(cen))
                nodes[str(ou)]["properties"]["gt_centroid_mm"] = cen
            if ("organ", o) in concept:
                edge(ou, "mapped_to_concept", concept[("organ", o)])
            if od.get("has_tumor"):
                n_les += 1
                lu = uri("lesion", cid, o, "tumor")
                node(lu, "Lesion", f"{o.replace('_', ' ')} tumor", is_tumor=True)
                typ(lu, "Lesion")
                tc = TUMOR_CONCEPT.get(o)
                if tc:
                    typ(lu, tc[0])
                lit(lu, "is_tumor", True, XSD.boolean)
                edge(ou, "has_lesion", lu)
                edge(lu, "located_in", ou)
                for key, gk, dt in [("tumor_volume_cm3", "gt_volume_cm3", XSD.float),
                                    ("tumor_voxels", "gt_voxels", XSD.integer),
                                    ("tumor_max_diameter_mm", "gt_max_diameter_mm", XSD.float),
                                    ("lesion_count", "lesion_count", XSD.integer)]:
                    val = od.get(key)
                    if val is not None:
                        lit(lu, gk, val, dt)
                        nodes[str(lu)]["properties"][gk] = val
                cen = od.get("tumor_centroid_mm")
                if cen:
                    lit(lu, "gt_centroid_mm", str(cen))
                    nodes[str(lu)]["properties"]["gt_centroid_mm"] = cen
                if ("tumor", o) in concept:
                    edge(lu, "mapped_to_concept", concept[("tumor", o)])
                for field, pred in OBS_PRED.items():
                    val = od.get(field)
                    if val in (None, "none", "na", "unknown", "pending"):
                        continue
                    lit(lu, pred, val)
                    nodes[str(lu)]["properties"][pred] = val

    g.serialize(f"{OUT}/imaging_kg_flare23.ttl", format="turtle")
    json.dump({"nodes": list(nodes.values()), "edges": edges, "schema": "kg/schema.owl",
               "datasets": ["flare"], "note": "FLARE23 GT-only enriched KG (organ+tumor, "
               "direct categorical triples, SNOMED/NCIt grounded)"},
              open(f"{OUT}/unified_mmkg_flare23.json", "w"))
    print(f"FLARE23 enriched: {n_cases} cases, {n_les} lesions, {len(g)} triples", flush=True)
    print("Nodes:", dict(Counter(n["type"] for n in nodes.values())), flush=True)
    print("Edges:", dict(Counter(e["relation"] for e in edges)), flush=True)


if __name__ == "__main__":
    main()
