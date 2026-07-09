# Imaging Phenotype KG Schema

OWL schema (T-Box) for the imaging knowledge graph that our SAM3 segmentation
hand-off populates. Matches the `acm_mmkg` conventions (`schema.py` v2.0):
namespace `http://example.org/mmkg/schema/`, Lesion semantic anchor, DICOM
Study/Series hierarchy, Anatomy/Organ classes, Observation phenotypes.

## Files
- `schema.owl` — RDF/XML OWL, opens in Protégé. Classes + object/datatype
  properties + `skos:exactMatch` links and `OntologyConcept` individuals.
- `ontology_mappings.json` — `Class::term → {system: {code, display, system}}`,
  same shape as `acm_mmkg/results/*/ontology_mappings.json`.
- `build_imaging_schema.py` (in `src/scripts/`) — regenerates both, fetching
  codes **live** from SNOMED CT (Snowstorm FHIR), NCIt (NCI EVS), FMA/RadLex (OLS4).

## Ontology mappings (verified)
Core anatomy/neoplasm codes are curated + directly `$lookup`-verified; softer
phenotype observations are live best-effort.

| Concept | SNOMED CT | NCIt |
|---|---|---|
| Pancreas | 15776009 (Pancreatic structure) | C12393 (Pancreas) |
| Liver | 10200004 (Liver structure) | C12392 (Liver) |
| Head of pancreas | 362201006 | — |
| Body of pancreas | 40133006 | — |
| Tail of pancreas | 73239005 | — |
| Pancreatic tumor | 372003004 (Primary malignant neoplasm) | C3305 (Pancreatic Neoplasm) |
| Liver tumor | 93870000 (Malignant neoplasm of liver) | C3099 (Hepatocellular Carcinoma) |
| Tumor diameter | 263605001 (Tumor size) | — |
| Tumor volume | 258261001 (Tumour volume) | — |
| Tumor burden | — | C28384 (Tumor Burden) |
| Cross-organ extension | 409771002 (Direct extension to adjacent organ) | — |

Unmapped (no clean single code — left blank rather than mis-coded): *Lesion
multiplicity*, *Organ containment*. FMA/RadLex not included: the OLS4 top-hits
were the wrong concept (e.g. "capillary network of islet"), so they were dropped.

## Model
```
ImagingCase ─has_study→ Study ─has_series→ Series
Series ─depicts_organ→ Organ (Pancreas|Liver)
Series ─has_lesion→ Lesion (PancreaticTumor|LiverTumor)
Lesion ─located_in→ AnatomicSite (Head|Body|Tail of pancreas) ─part_of→ Organ
Lesion ─has_observation→ Observation (diameter|volume|burden|multiplicity|containment|cross-organ)
Lesion ─overlaps_organ→ Organ            (containment / cross-organ signal)
<any> ─mapped_to_concept→ OntologyConcept (code + system + display + IRI)
```
Datatype props: `dataset, modality, case_id, volume_cm3, diameter_mm,
lesion_count, anatomic_location, dice_score, is_tumor, source_of_segmentation`.

Regenerate: `python src/scripts/build_imaging_schema.py`
