#!/usr/bin/env python3
"""
kg_query.py — query interface over the persisted imaging KG (kg/graph/unified_mmkg.json).

Fast Python-side graph traversal (rdflib's SPARQL engine is too slow for multi-hop joins on
this size). Exposes a small KG API + a set of named complex queries that span
ImagingCase -> Organ -> Lesion -> Observation with ontology grounding and anatomic sub-sites.

  python kg_query.py                # list + run all named queries
  python kg_query.py <name>         # run one (e.g. high_burden_pancreatic_tumor)
Import:  from kg_query import KG ;  kg = KG();  kg.cases(...)
"""
import json, sys
from collections import defaultdict

GRAPH = "/home/ud3d4/Desktop/SWOG/kg/graph/unified_mmkg.json"


class KG:
    def __init__(self, path=GRAPH):
        G = json.load(open(path))
        self.nodes = {n["id"]: n for n in G["nodes"]}
        self.out, self.inn = defaultdict(list), defaultdict(list)
        for e in G["edges"]:
            self.out[(e["source"], e["relation"])].append(e["target"])
            self.inn[(e["target"], e["relation"])].append(e["source"])
        self.cases_all = [n["id"] for n in G["nodes"] if n["type"] == "ImagingCase"]
        self.patients = [n["id"] for n in G["nodes"] if n["type"] == "Patient"]

    # ---- primitives ----
    def objs(self, s, rel):
        return self.out[(s, rel)]

    def label(self, nid):
        return self.nodes[nid].get("label")

    def prop(self, nid, k):
        return self.nodes[nid]["properties"].get(k)

    def organ_lesions(self, case, organ_label=None):
        for o in self.objs(case, "depicts_organ"):
            if organ_label and self.label(o) != organ_label:
                continue
            for l in self.objs(o, "has_lesion"):
                yield o, l

    def obs_value(self, lesion, obstype):
        for ob in self.objs(lesion, "has_observation"):
            if self.nodes[ob]["type"] == obstype:
                return self.prop(ob, "value")
        return None

    def at_site(self, lesion, site_prefix):
        return any((self.label(s) or "").startswith(site_prefix) for s in self.objs(lesion, "at_site"))

    def concept_by_code(self, code):
        return next((n["id"] for n in self.nodes.values()
                     if n["type"] == "OntologyConcept" and n["properties"].get("code") == code), None)

    def cases(self, pred):
        return sorted(c for c in self.cases_all if pred(c))

    # ---- patient-level ----
    def patient_of(self, case):
        ps = self.objs(case, "of_patient")
        return ps[0] if ps else None

    def is_patient_level(self, case):
        return bool(self.objs(case, "of_patient"))

    def patients_where(self, pred):
        """Patients (via their case) satisfying pred — only per-patient (volume) cases count."""
        return sorted({self.patient_of(c) for c in self.cases_all
                       if self.is_patient_level(c) and pred(c)})

    def n_organs(self, case):
        return len(self.objs(case, "depicts_organ"))


# ---- named complex queries ----
def QUERIES(kg):
    return {
        # -- patient-level --
        "patients_total": kg.patients,
        "patients_flare": kg.patients_where(lambda c: kg.prop(c, "dataset") == "flare"),
        "patients_observing_pancreas": kg.patients_where(lambda c: any(kg.label(o) == "pancreas" for o in kg.objs(c, "depicts_organ"))),
        "patients_multi_organ_ge2": kg.patients_where(lambda c: kg.n_organs(c) >= 2),
        "patients_with_pancreatic_tumor": kg.patients_where(lambda c: any(True for _ in kg.organ_lesions(c, "pancreas"))),
        "patients_high_burden_any_tumor": kg.patients_where(lambda c: any(kg.obs_value(l, "TumorBurden") == "high" for _, l in kg.organ_lesions(c))),
        # -- phenotype (patient-level now that all cases are patients) --
        "high_burden_pancreatic_tumor":
            kg.cases(lambda c: any(kg.obs_value(l, "TumorBurden") == "high"
                                   for _, l in kg.organ_lesions(c, "pancreas"))),
        "cross_organ_tumor":
            kg.cases(lambda c: any(kg.objs(l, "overlaps_organ") for _, l in kg.organ_lesions(c))),
        "multifocal_liver_tumor":
            kg.cases(lambda c: any(kg.obs_value(l, "LesionMultiplicity") == "multifocal"
                                   for _, l in kg.organ_lesions(c, "liver"))),
        "contained_pancreatic_head_tumor":
            kg.cases(lambda c: any(kg.obs_value(l, "OrganContainment") == "contained" and kg.at_site(l, "head")
                                   for _, l in kg.organ_lesions(c, "pancreas"))),
        "solitary_high_burden_liver":
            kg.cases(lambda c: any(kg.obs_value(l, "TumorBurden") == "high" and
                                   kg.obs_value(l, "LesionMultiplicity") == "solitary"
                                   for _, l in kg.organ_lesions(c, "liver"))),
        "grounded_liver_malignancy_SNOMED_93870000":
            [kg.nodes[s] and s for s in kg.inn[(kg.concept_by_code("93870000"), "mapped_to_concept")]],
    }


def main():
    kg = KG()
    print(f"KG: {len(kg.nodes)} nodes, {len(kg.cases_all)} ImagingCases\n")
    qs = QUERIES(kg)
    want = sys.argv[1:] or list(qs)
    for name in want:
        if name not in qs:
            print(f"  unknown query '{name}'. available: {', '.join(qs)}"); continue
        res = qs[name]
        ex = [r.replace("case_", "") for r in res[:3] if isinstance(r, str)]
        print(f"{name:42s} -> {len(res):4d}   e.g. {ex}")


if __name__ == "__main__":
    main()
