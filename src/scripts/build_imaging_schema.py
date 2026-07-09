#!/usr/bin/env python3
"""
Build an OWL schema (schema.owl) for the imaging phenotype KG that our SAM3
segmentation hand-off populates, with ontology mappings fetched LIVE from:
  - SNOMED CT   (Snowstorm FHIR $expand)      -- anatomy, findings
  - NCIt        (NCI EVS REST)                -- neoplasms, cancer terms
  - FMA / RadLex(EBI OLS4)                    -- anatomy (FMA), imaging (RadLex)

Matches the acm_mmkg conventions (schema.py v2.0): namespace
http://example.org/mmkg/schema/, Lesion semantic anchor, DICOM Study/Series,
Anatomy->FMA, Observation->RadLex, Disease->SNOMED/NCIt. Emits:
  - schema.owl                 (RDF/XML, T-Box + skos:exactMatch to ontology IRIs)
  - ontology_mappings.json     (Class::term -> {system: {code, display, system}})
"""
import os, json, time
import requests
from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, OWL, XSD, SKOS, DCTERMS

OUT_DIR = "/home/ud3d4/Desktop/SWOG/kg"
os.makedirs(OUT_DIR, exist_ok=True)
MM = Namespace("http://example.org/mmkg/schema/")

SYS_URL = {
    "SNOMED CT": "https://www.snomed.org/snomed-ct",
    "NCIt": "https://ncithesaurus.nci.nih.gov",
    "FMA": "http://si.washington.edu/projects/fma",
    "RadLex": "https://www.rsna.org/practice-tools/data-tools-and-standards/radlex-radiology-lexicon",
}


# ── live ontology lookups ──
def snomed(term):
    try:
        r = requests.get("https://snowstorm-training.snomedtools.org/fhir/ValueSet/$expand",
                         params={"url": "http://snomed.info/sct?fhir_vs", "filter": term, "count": 5}, timeout=30)
        items = r.json().get("expansion", {}).get("contains", [])
        if items:
            it = items[0]
            return {"system": "SNOMED CT", "code": it["code"], "display": it["display"],
                    "iri": f"http://snomed.info/id/{it['code']}"}
    except Exception as e:
        print(f"  SNOMED miss '{term}': {e}")
    return None


def ncit(term):
    try:
        r = requests.get("https://api-evsrest.nci.nih.gov/api/v1/concept/ncit/search",
                         params={"term": term, "pageSize": 5, "type": "contains"}, timeout=30)
        cs = r.json().get("concepts", [])
        if cs:
            c = cs[0]
            return {"system": "NCIt", "code": c["code"], "display": c["name"],
                    "iri": f"http://purl.obolibrary.org/obo/NCIT_{c['code']}"}
    except Exception as e:
        print(f"  NCIt miss '{term}': {e}")
    return None


def ols(term, ontology):
    try:
        r = requests.get("https://www.ebi.ac.uk/ols4/api/search",
                         params={"q": term, "ontology": ontology, "rows": 5, "exact": "false"}, timeout=30)
        docs = r.json().get("response", {}).get("docs", [])
        for d in docs:
            obo = d.get("obo_id") or ""
            if obo:
                sysname = "FMA" if ontology == "fma" else "RadLex"
                return {"system": sysname, "code": obo, "display": d.get("label"),
                        "iri": d.get("iri") or f"http://purl.obolibrary.org/obo/{obo.replace(':', '_')}"}
    except Exception as e:
        print(f"  {ontology} miss '{term}': {e}")
    return None


# Curated, directly-verified canonical codes for the core anatomy / neoplasm concepts.
# (Auto "top hit" search is unreliable for these — verified via FHIR/EVS $lookup.)
def _iri(system, code):
    return (f"http://snomed.info/id/{code}" if system == "SNOMED CT"
            else f"http://purl.obolibrary.org/obo/NCIT_{code}")

CURATED = {
    "Pancreas": {"SNOMED CT": ("15776009", "Pancreatic structure"), "NCIt": ("C12393", "Pancreas")},
    "Liver": {"SNOMED CT": ("10200004", "Liver structure"), "NCIt": ("C12392", "Liver")},
    "PancreaticHead": {"SNOMED CT": ("362201006", "Entire head of pancreas")},
    "PancreaticBody": {"SNOMED CT": ("40133006", "Structure of body of pancreas")},
    "PancreaticTail": {"SNOMED CT": ("73239005", "Structure of tail of pancreas")},
    "PancreaticTumor": {"SNOMED CT": ("372003004", "Primary malignant neoplasm of pancreas"),
                        "NCIt": ("C3305", "Pancreatic Neoplasm")},
    "LiverTumor": {"SNOMED CT": ("93870000", "Malignant neoplasm of liver"),
                   "NCIt": ("C3099", "Hepatocellular Carcinoma")},
}
STOP = {"organ", "structure", "entire", "part", "region", "the", "of", "and"}


def _sane(label, display):
    """Accept a live match only if its display shares a content word with the label."""
    lw = {w for w in label.lower().replace("-", " ").split() if w not in STOP}
    dw = {w for w in display.lower().replace("-", " ").split()}
    return bool(lw & dw)


def fetch(local, label, term, systems):
    out = {}
    if "SNOMED" in systems:
        m = snomed(term)
        if m and _sane(label, m["display"]):
            out["SNOMED CT"] = m
    if "NCIt" in systems:
        m = ncit(term)
        if m and _sane(label, m["display"]):
            out["NCIt"] = m
    if "FMA" in systems:
        m = ols(term, "fma")
        if m and _sane(label, m["display"]):
            out["FMA"] = m
    if "RadLex" in systems:
        m = ols(term, "radlex")
        if m and _sane(label, m["display"]):
            out["RadLex"] = m
    # curated codes are authoritative — overlay them
    for sysname, (code, disp) in CURATED.get(local, {}).items():
        out[sysname] = {"system": sysname, "code": code, "display": disp, "iri": _iri(sysname, code)}
    time.sleep(0.15)
    return out


# ── imaging phenotype vocabulary (node_class, class_local, label, query_term, systems) ──
CONCEPTS = [
    ("Organ", "Pancreas", "Pancreas", "pancreas structure", ["SNOMED", "NCIt", "FMA", "RadLex"]),
    ("Organ", "Liver", "Liver", "liver structure", ["SNOMED", "NCIt", "FMA", "RadLex"]),
    ("AnatomicSite", "PancreaticHead", "Head of pancreas", "head of pancreas", ["SNOMED"]),
    ("AnatomicSite", "PancreaticBody", "Body of pancreas", "body of pancreas", ["SNOMED"]),
    ("AnatomicSite", "PancreaticTail", "Tail of pancreas", "tail of pancreas", ["SNOMED"]),
    ("Lesion", "PancreaticTumor", "Pancreatic tumor", "malignant tumor of pancreas", ["SNOMED", "NCIt"]),
    ("Lesion", "LiverTumor", "Liver tumor", "malignant tumor of liver", ["SNOMED", "NCIt"]),
    ("Observation", "TumorDiameter", "Tumor diameter", "tumor size", ["SNOMED", "RadLex"]),
    ("Observation", "TumorVolume", "Tumor volume", "tumor volume", ["SNOMED", "RadLex"]),
    ("Observation", "TumorBurden", "Tumor burden", "tumor burden", ["SNOMED", "NCIt"]),
    ("Observation", "LesionMultiplicity", "Lesion multiplicity", "multiple masses", ["SNOMED", "RadLex"]),
    ("Observation", "OrganContainment", "Organ containment", "tumor confined to organ", ["SNOMED", "NCIt"]),
    ("Observation", "CrossOrganExtension", "Cross-organ extension", "tumor invasion of adjacent organ", ["SNOMED", "NCIt"]),
]

# top-level classes and their parents
CLASSES = {
    "ImagingCase": None, "Study": None, "Series": None, "OntologyConcept": None,
    "Organ": "Anatomy", "AnatomicSite": "Anatomy", "Anatomy": None,
    "Lesion": None, "Observation": None,
}
OBJ_PROPS = [
    ("has_study", "ImagingCase", "Study"), ("has_series", "Study", "Series"),
    ("depicts_organ", "Series", "Organ"), ("has_lesion", "Series", "Lesion"),
    ("located_in", "Lesion", "AnatomicSite"), ("part_of", "AnatomicSite", "Organ"),
    ("has_observation", "Lesion", "Observation"), ("overlaps_organ", "Lesion", "Organ"),
    ("mapped_to_concept", None, "OntologyConcept"),
]
DATA_PROPS = [
    ("dataset", "string"), ("modality", "string"), ("case_id", "string"),
    ("volume_cm3", "float"), ("diameter_mm", "float"), ("lesion_count", "integer"),
    ("anatomic_location", "string"), ("dice_score", "float"), ("is_tumor", "boolean"),
    ("source_of_segmentation", "string"),
]
XSD_OF = {"string": XSD.string, "float": XSD.float, "integer": XSD.integer, "boolean": XSD.boolean}


def build():
    print("Fetching ontology mappings (live)...")
    mappings = {}
    concept_maps = {}
    for node_class, local, label, term, systems in CONCEPTS:
        m = fetch(local, label, term, systems)
        concept_maps[local] = m
        mappings[f"{node_class}::{label}"] = {sys: {k: v for k, v in d.items() if k != "iri"} for sys, d in m.items()}
        got = ",".join(m.keys()) or "none"
        print(f"  {label:24s} -> {got}")

    g = Graph()
    for pfx, ns in [("owl", OWL), ("rdfs", RDFS), ("skos", SKOS), ("dcterms", DCTERMS), ("mm", MM)]:
        g.bind(pfx, ns)

    onto = URIRef("http://example.org/mmkg/schema")
    g.add((onto, RDF.type, OWL.Ontology))
    g.add((onto, DCTERMS.title, Literal("Imaging Phenotype KG Schema (SAM3 segmentation hand-off)")))
    g.add((onto, RDFS.comment, Literal(
        "T-Box for imaging cases, organs, lesions and phenotypes derived from SAM3 segmentation. "
        "Domain classes carry skos:exactMatch to SNOMED CT / NCIt / FMA / RadLex concepts. "
        "Aligned with acm_mmkg schema.py v2.0 conventions.")))
    g.add((onto, OWL.versionInfo, Literal("1.0")))

    # classes
    for cls, parent in CLASSES.items():
        c = MM[cls]
        g.add((c, RDF.type, OWL.Class)); g.add((c, RDFS.label, Literal(cls)))
        if parent:
            g.add((c, RDFS.subClassOf, MM[parent]))
    # object properties
    for name, dom, rng in OBJ_PROPS:
        p = MM[name]; g.add((p, RDF.type, OWL.ObjectProperty)); g.add((p, RDFS.label, Literal(name)))
        if dom: g.add((p, RDFS.domain, MM[dom]))
        if rng: g.add((p, RDFS.range, MM[rng]))
    # datatype properties
    for name, t in DATA_PROPS:
        p = MM[name]; g.add((p, RDF.type, OWL.DatatypeProperty)); g.add((p, RDFS.label, Literal(name)))
        g.add((p, RDFS.range, XSD_OF[t]))

    # annotation props for the concept codes
    for ap in ("code", "system", "display"):
        g.add((MM[ap], RDF.type, OWL.AnnotationProperty)); g.add((MM[ap], RDFS.label, Literal(ap)))

    # domain subclasses per concept + ontology mappings
    for node_class, local, label, term, systems in CONCEPTS:
        c = MM[local]
        g.add((c, RDF.type, OWL.Class)); g.add((c, RDFS.label, Literal(label)))
        g.add((c, RDFS.subClassOf, MM[node_class]))
        for sysname, d in concept_maps[local].items():
            # link the class to the real ontology IRI
            g.add((c, SKOS.exactMatch, URIRef(d["iri"])))
            # also materialize an OntologyConcept individual (their pattern)
            oc = MM[f"concept_{local}_{d['code'].replace(':', '_')}"]
            g.add((oc, RDF.type, MM.OntologyConcept))
            g.add((oc, MM.code, Literal(d["code"])))
            g.add((oc, MM.system, Literal(sysname)))
            g.add((oc, MM.display, Literal(d["display"])))
            g.add((oc, RDFS.seeAlso, URIRef(d["iri"])))
            g.add((oc, RDFS.isDefinedBy, URIRef(SYS_URL[sysname])))
            g.add((c, MM.mapped_to_concept, oc))

    owl_path = os.path.join(OUT_DIR, "schema.owl")
    g.serialize(owl_path, format="pretty-xml")
    json.dump({"mappings": mappings,
               "stats": {"concepts": len(CONCEPTS),
                         "by_system": {s: sum(1 for m in mappings.values() if s in m)
                                       for s in ["SNOMED CT", "NCIt", "FMA", "RadLex"]}}},
              open(os.path.join(OUT_DIR, "ontology_mappings.json"), "w"), indent=2)
    print(f"\nschema.owl: {owl_path}  ({len(g)} triples)")
    print(f"ontology_mappings.json: {os.path.join(OUT_DIR, 'ontology_mappings.json')}")


if __name__ == "__main__":
    build()
