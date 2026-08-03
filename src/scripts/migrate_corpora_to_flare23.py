#!/usr/bin/env python3
"""Migrate the app + demo-notebook corpora from the 100-case FLARE22 DEMO to the full 1,312-patient
FLARE23. Rebuilds corpus_perpatient.json and corpus_global.json: keep Pancreas (281) + LiTS (131)
[+ ingested], swap the 100 FLARE22_Tr records for the 1,312 FLARE23 records from corpus_flare_train.json.
Originals are backed up to kg/data/_archive_demo_flare/ first (reversible)."""
import json
import os
import shutil
from collections import Counter

D = "/home/ud3d4/Desktop/SWOG/kg/data"
ARC = f"{D}/_archive_demo_flare"
os.makedirs(ARC, exist_ok=True)

flare23 = json.load(open(f"{D}/corpus_flare_train.json"))["records"]          # 1,312 FLARE23_xxxx
assert all(r["dataset"] == "flare" for r in flare23), "corpus_flare_train must be all-flare"
print(f"FLARE23 (full) records to splice in: {len(flare23)}")

for name in ["corpus_perpatient.json", "corpus_global.json"]:
    path = f"{D}/{name}"
    obj = json.load(open(path))
    recs = obj["records"] if isinstance(obj, dict) and "records" in obj else obj
    before = Counter(r.get("dataset") for r in recs)
    kept = [r for r in recs if r.get("dataset") != "flare"]                    # keep pancreas/lits/ingested
    new = kept + flare23
    after = Counter(r.get("dataset") for r in new)
    shutil.copy(path, f"{ARC}/{name}")                                        # backup original (demo)
    out = {"records": new} if isinstance(obj, dict) and "records" in obj else new
    json.dump(out, open(path, "w"))
    print(f"{name}: {dict(before)} -> {dict(after)}  (backup -> _archive_demo_flare/{name})")

print("done.")
