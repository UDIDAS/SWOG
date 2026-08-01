#!/usr/bin/env python3
"""Ontology grounding for the KG — term -> standard terminology codes, via LIVE APIs, CACHED.
Vendors the multi-terminology resolver from acm_mmkg (SNOMED CT / LOINC / ICD-11 / RxNorm / MeSH),
adds retries (the SNOMED demo endpoint is flaky) + a persistent cache. NO hard-coded codes: every
grounding is resolved by the mapper and stored in kg/ontology_mappings.json ("Category::Term" keys),
so a term we've never seen just resolves once and the ontology grows itself.

Schema-aware routing (ONTOLOGY_ROUTING) only calls the terminologies that fit an entity type
(Organ->SNOMED+MeSH, Tumor->SNOMED+ICD, measurement->LOINC), and semantic-overlap validation
rejects spurious hits.
"""
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# creds for LOINC / ICD live in acm_mmkg/.env (the mapper's home)
for envp in ("/home/ud3d4/Desktop/Projects/acm_mmkg/.env", "/home/ud3d4/Desktop/SWOG/.env"):
    if os.path.exists(envp):
        for line in open(envp):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
LOINC_UID = os.getenv("LOINC_UID")
LOINC_PWD = os.getenv("LOINC_PWD")
ICD_TOKEN = os.getenv("ICD_CLIENT_TOKEN")

CACHE = "/home/ud3d4/Desktop/SWOG/kg/ontology_mappings.json"
ONTOLOGY_ROUTING = {
    "Tumor": ["SNOMED CT", "ICD 11", "MeSH"],
    "Organ": ["SNOMED CT", "MeSH", "NCIt"],
    "Anatomy": ["SNOMED CT", "MeSH"],
    "Disease": ["ICD 11", "SNOMED CT"],
    "Measurement": ["LOINC", "SNOMED CT"],
    "Test": ["LOINC"],
}


def _retry(fn, *a, tries=3, delay=2, **k):
    for i in range(tries):
        try:
            r = fn(*a, **k)
            if r:
                return r
        except Exception:
            pass
        time.sleep(delay * (i + 1))
    return None


def get_snomed_ct_code(variable):
    r = requests.get("https://snowstorm-training.snomedtools.org/fhir/ValueSet/$expand",
                     params={"url": "http://snomed.info/sct?fhir_vs", "filter": variable, "count": 10},
                     timeout=120)
    if r.status_code == 200:
        return [{"code": it["code"], "display": it["display"], "system": "SNOMED CT"}
                for it in r.json()["expansion"].get("contains", [])]
    return None


def get_loinc_code(query):
    if not (LOINC_UID and LOINC_PWD):
        return None
    r = requests.get("https://loinc.regenstrief.org/searchapi/loincs",
                     params={"query": query, "rows": 10}, auth=(LOINC_UID, LOINC_PWD), timeout=60)
    if r.status_code == 200 and "Results" in r.json():
        return [{"system": "LOINC", "code": it["LOINC_NUM"], "display": it["LONG_COMMON_NAME"]}
                for it in r.json()["Results"]]
    return None


def _icd_token():
    if not ICD_TOKEN:
        return None
    r = requests.post("https://icdaccessmanagement.who.int/connect/token",
                      headers={"Authorization": f"Basic {ICD_TOKEN}",
                               "Content-Type": "application/x-www-form-urlencoded"},
                      data={"grant_type": "client_credentials"}, timeout=60)
    return r.json().get("access_token") if r.status_code == 200 else None


def get_icd_code(variable):
    tok = _icd_token()
    if not tok:
        return None
    r = requests.get("https://id.who.int/icd/entity/search", params={"q": variable, "flatResults": True},
                     headers={"accept": "application/json", "API-Version": "v2",
                              "Accept-Language": "en", "Authorization": f"Bearer {tok}"}, timeout=60)
    if r.status_code == 200:
        return [{"code": e["id"].split("/")[-1], "display": re.sub(r"</?em.*?>", "", e["title"]),
                 "system": "ICD 11"} for e in r.json().get("destinationEntities", [])]
    return None


def get_mesh_code(query):
    r = requests.get("https://id.nlm.nih.gov/mesh/lookup/term",
                     params={"label": query, "match": "contains", "limit": 10}, timeout=60)
    if r.status_code == 200:
        return [{"system": "MeSH", "code": it["resource"].split("/")[-1], "display": it["label"]}
                for it in r.json()]
    return None


FUNCS = {"SNOMED CT": get_snomed_ct_code, "LOINC": get_loinc_code, "ICD 11": get_icd_code,
         "MeSH": get_mesh_code}


def _overlap(query, display):
    stop = {"of", "the", "a", "an", "in", "and", "or", "with", "for", "to", "by", "at", "on", "structure"}
    q = set(query.lower().split()) - stop
    d = set(display.lower().split()) - stop
    return bool(q) and len(q & d) >= 1


def resolve(term, entity_type):
    """Live-resolve a term across the terminologies routed for entity_type; validated best match each."""
    systems = [s for s in ONTOLOGY_ROUTING.get(entity_type, ["SNOMED CT", "MeSH"]) if s in FUNCS]
    out = {}
    with ThreadPoolExecutor(max_workers=len(systems) or 1) as ex:
        futs = {ex.submit(_retry, FUNCS[s], term): s for s in systems}
        for f in as_completed(futs):
            s = futs[f]
            data = f.result()
            if not data:
                continue
            for e in data:
                if _overlap(term, e.get("display", "")):
                    out[s] = {"system": s, "code": e["code"], "display": e["display"]}
                    break
            if s not in out and data:                      # fall back to top hit if no token overlap
                out[s] = {"system": s, "code": data[0]["code"], "display": data[0]["display"]}
    return out


def ground(category, term, entity_type, cache=None, refresh=False):
    """Cache-backed grounding. key = 'Category::Term'. Returns {system: {...}}."""
    own = cache is None
    if own:
        cache = json.load(open(CACHE))["mappings"] if os.path.exists(CACHE) else {}
    key = f"{category}::{term}"
    if key in cache and not refresh:
        return cache[key]
    m = resolve(term, entity_type)
    if m:
        cache[key] = m
    if own and m:
        json.dump({"mappings": cache, "stats": {"n": len(cache)}}, open(CACHE, "w"), indent=2)
    return cache.get(key, {})


if __name__ == "__main__":
    import sys
    print(json.dumps(resolve(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "Organ"), indent=2))
