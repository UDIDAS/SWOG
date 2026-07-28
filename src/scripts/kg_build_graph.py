#!/usr/bin/env python3
"""
kg_build_graph.py — OBTAIN the persisted imaging knowledge graph.

Materializes the KG from the schema + ontology mappings + phenotype instances:
  in:  kg/schema.owl              (ontology: classes + relations)
       kg/ontology_mappings.json  (concept -> SNOMED/NCIt grounding)
       kg/data/corpus_3regime.json (1010 ImagingCase instances: Pancreas+LiTS+FLARE)
  out: kg/graph/imaging_kg.ttl    (canonical RDF/Turtle graph, includes the ontology)
       kg/graph/unified_mmkg.json (nodes[]+edges[] graph for easy load / app ingestion)

Nodes: ImagingCase, Dataset, Organ, Lesion, OntologyConcept, AnatomicSite, Observation.
Edges: from_dataset, depicts_organ, has_lesion, located_in, overlaps_organ, mapped_to_concept,
       has_observation, at_site.  Grounds organs/lesions/sub-sites to SNOMED CT / NCIt.
"""
import json, os, sys
from collections import Counter
from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS, XSD

KG = "/home/ud3d4/Desktop/SWOG/kg"
DATA = f"{KG}/data"
OUT = f"{KG}/graph"
# default = PER-PATIENT corpus (patient-level queries). pass corpus_3regime.json for slice-level.
CORPUS = sys.argv[1] if len(sys.argv) > 1 else f"{DATA}/corpus_perpatient.json"
TAG = "" if "perpatient" in os.path.basename(CORPUS) else \
    "_" + os.path.basename(CORPUS).replace("corpus_", "").replace(".json", "")
MMKG = Namespace("http://example.org/mmkg/schema/")
INST = Namespace("http://example.org/mmkg/instance/")

ORGAN_MAPKEY = {"liver": "Organ::Liver", "pancreas": "Organ::Pancreas", "spleen": "Organ::Spleen",
                "right_kidney": "Organ::Right kidney", "left_kidney": "Organ::Left kidney"}
ORGAN_CLASS = {"liver": "Liver", "pancreas": "Pancreas", "spleen": "Spleen",
               "right_kidney": "Kidney", "left_kidney": "Kidney"}
TUMOR_MAPKEY = {"pancreas": "Lesion::Pancreatic tumor", "liver": "Lesion::Liver tumor"}
TUMOR_CLASS = {"pancreas": "PancreaticTumor", "liver": "LiverTumor"}
SITE_MAPKEY = {"head": "AnatomicSite::Head of pancreas", "body": "AnatomicSite::Body of pancreas",
               "tail": "AnatomicSite::Tail of pancreas"}
SITE_CLASS = {"head": "PancreaticHead", "body": "PancreaticBody", "tail": "PancreaticTail"}
OBS = {"burden_cat": ("TumorBurden", "Observation::Tumor burden"),
       "multiplicity": ("LesionMultiplicity", "Observation::Lesion multiplicity"),
       "containment": ("OrganContainment", "Observation::Organ containment")}
# standard SNOMED/NCIt for FLARE organs absent from the paper's ontology_mappings.json
SUPP = {
    "Organ::Spleen": {"SNOMED CT": ("78961009", "Splenic structure"), "NCIt": ("C12432", "Spleen")},
    "Organ::Right kidney": {"SNOMED CT": ("9846003", "Right kidney structure")},
    "Organ::Left kidney": {"SNOMED CT": ("18639004", "Left kidney structure")},
}


def uri(*parts):
    return INST["_".join(str(p).replace(" ", "").replace(":", "").replace("/", "") for p in parts)]


def main():
    os.makedirs(OUT, exist_ok=True)
    g = Graph()
    g.parse(f"{KG}/schema.owl")                       # KG includes the ontology
    g.bind("mmkg", MMKG); g.bind("inst", INST)
    nodes, edges = {}, []                              # parallel unified-JSON graph

    def node(nid, ntype, label=None, **props):
        n = nodes.setdefault(str(nid), {"id": str(nid).split("/")[-1], "type": ntype, "properties": {}})
        if label:
            n["label"] = label
        n["properties"].update({k: v for k, v in props.items() if v is not None})
        return nid

    def edge(s, rel, o):
        g.add((s, MMKG[rel], o)); edges.append({"source": str(s).split("/")[-1], "relation": rel, "target": str(o).split("/")[-1]})

    def typ(u, cls):
        g.add((u, RDF.type, MMKG[cls]))

    def lit(u, prop, val, dt=None):
        g.add((u, MMKG[prop], Literal(val, datatype=dt) if dt else Literal(val)))

    # ---- ontology concepts (verified mappings + supplemental) ----
    mappings = json.load(open(f"{KG}/ontology_mappings.json"))["mappings"]
    concept_uri = {}                                   # mapkey -> concept node uri (first system)
    def add_concepts(mapkey, systems):
        first = None
        for sysname, d in systems.items():
            code = d["code"] if isinstance(d, dict) else d[0]
            disp = d.get("display") if isinstance(d, dict) else d[1]
            if not code:
                continue
            cu = uri("concept", sysname, code)
            node(cu, "OntologyConcept", disp, system=sysname, code=code); typ(cu, "OntologyConcept")
            lit(cu, "system", sysname); lit(cu, "code", code); lit(cu, "display", disp)
            first = first or cu
        if first:
            concept_uri[mapkey] = first
    for mk, systems in mappings.items():
        add_concepts(mk, systems)
    for mk, systems in SUPP.items():
        add_concepts(mk, {s: {"code": c, "display": disp} for s, (c, disp) in systems.items()})

    # ---- instances ----
    recs = json.load(open(CORPUS))["records"]
    for r in recs:
        cid = r["case_id"]; cu = uri("case", cid); gran = r.get("granularity")
        node(cu, "ImagingCase", cid, dataset=r["dataset"], granularity=gran, modality="CT")
        typ(cu, "ImagingCase"); lit(cu, "case_id", cid); lit(cu, "dataset", r["dataset"])
        lit(cu, "modality", "CT"); lit(cu, "granularity", gran)
        du = uri("dataset", r["dataset"]); node(du, "Dataset", r["dataset"]); typ(du, "Dataset")
        edge(cu, "from_dataset", du)
        # PATIENT layer: a per-patient (volume) case IS a patient study -> patient-level queries
        if gran == "volume":
            pu = uri("patient", cid)
            node(pu, "Patient", cid, patient_id=cid, dataset=r["dataset"]); typ(pu, "Patient")
            lit(pu, "patient_id", cid); edge(cu, "of_patient", pu)

        tumor_organs = [o for o in r["observed_organs"] if r["organs"][o].get("has_tumor")]
        for o in r["observed_organs"]:
            od = r["organs"][o]; ou = uri("organ", cid, o)
            vol = od.get("organ_volume_cm3"); area = od.get("organ_area_px")
            node(ou, "Organ", o.replace("_", " "), volume_cm3=vol, area_px=area)
            typ(ou, "Organ"); typ(ou, ORGAN_CLASS.get(o, "Organ")); lit(ou, "display", o.replace("_", " "))
            if vol is not None:
                lit(ou, "volume_cm3", vol, XSD.float)
            edge(cu, "depicts_organ", ou)
            if ORGAN_MAPKEY.get(o) in concept_uri:
                edge(ou, "mapped_to_concept", concept_uri[ORGAN_MAPKEY[o]])

            if od.get("has_tumor"):
                lu = uri("lesion", cid, o)
                node(lu, "Lesion", f"tumor in {o.replace('_',' ')}", is_tumor=True,
                     volume_cm3=od.get("tumor_volume_cm3"), area_px=od.get("tumor_area_px"))
                typ(lu, "Lesion"); typ(lu, TUMOR_CLASS.get(o, "Lesion")); lit(lu, "is_tumor", True, XSD.boolean)
                if od.get("tumor_volume_cm3") is not None:
                    lit(lu, "volume_cm3", od["tumor_volume_cm3"], XSD.float)
                edge(ou, "has_lesion", lu); edge(lu, "located_in", ou)
                if TUMOR_MAPKEY.get(o) in concept_uri:
                    edge(lu, "mapped_to_concept", concept_uri[TUMOR_MAPKEY[o]])
                # observations (burden / multiplicity / containment)
                for field, (cls, mk) in OBS.items():
                    val = od.get(field)
                    if val in (None, "none", "na", "unknown"):
                        continue
                    obu = uri("obs", cid, o, field)
                    node(obu, cls, f"{field}={val}", value=val); typ(obu, cls); lit(obu, "value", val)
                    edge(lu, "has_observation", obu)
                    if mk in concept_uri:
                        edge(obu, "mapped_to_concept", concept_uri[mk])
                # anatomic sub-site (pancreas head/body/tail)
                loc = od.get("anatomic_location")
                if loc in SITE_MAPKEY:
                    su = uri("site", loc); node(su, "AnatomicSite", f"{loc} of pancreas")
                    typ(su, SITE_CLASS[loc]); edge(lu, "at_site", su)
                    if SITE_MAPKEY[loc] in concept_uri:
                        edge(su, "mapped_to_concept", concept_uri[SITE_MAPKEY[loc]])
                # cross-organ extension
                if r.get("cross_organ"):
                    for o2 in tumor_organs:
                        if o2 != o:
                            edge(lu, "overlaps_organ", uri("organ", cid, o2))

    # ---- persist ----
    g.serialize(f"{OUT}/imaging_kg{TAG}.ttl", format="turtle")
    unified = {"nodes": list(nodes.values()), "edges": edges,
               "schema": "kg/schema.owl", "namespace": str(INST), "corpus": os.path.basename(CORPUS)}
    json.dump(unified, open(f"{OUT}/unified_mmkg{TAG}.json", "w"))

    # ---- report ----
    ntypes = Counter(n["type"] for n in nodes.values())
    etypes = Counter(e["relation"] for e in edges)
    npat = ntypes.get("Patient", 0)
    print(f"KG built from {len(recs)} ImagingCases ({npat} patient-level) [{os.path.basename(CORPUS)}]"
          f" -> {OUT}/imaging_kg{TAG}.ttl + unified_mmkg{TAG}.json")
    print(f"Triples: {len(g)}")
    print(f"Nodes ({len(nodes)}): " + ", ".join(f"{k}={v}" for k, v in ntypes.most_common()))
    print(f"Edges ({len(edges)}): " + ", ".join(f"{k}={v}" for k, v in etypes.most_common()))
    print(f"Grounded concepts: {len(concept_uri)}")


if __name__ == "__main__":
    main()
