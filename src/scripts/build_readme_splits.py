#!/usr/bin/env python3
"""Fill the per-dataset train/val/test split tables in README.md between the <!-- SPLITS:START/END --> markers.
Computed the SAME way training splits (filter to one dataset, then patient_split with RandomState(42),
20% test / 10% val by whole patient) — from the pool meta.json files only (no image arrays). Verified to
reproduce the tr/va/te slice counts in the training logs exactly. CPU-only, re-runnable.
"""
import json
import numpy as np
from collections import defaultdict

README = "/home/ud3d4/Desktop/SWOG/README.md"
START, END = "<!-- SPLITS:START -->", "<!-- SPLITS:END -->"
ORGAN_META = "/scratch/ud3d4/acm_data/organ_pool_lkp/meta.json"
TUMOR_METAS = ["/scratch/ud3d4/acm_data/tumor_pool/meta.json",
               "/scratch/ud3d4/acm_data/flare_tumor_pool/meta.json",
               "/scratch/ud3d4/acm_data/kits_tumor_pool/meta.json"]
ORGAN_DS = [("LiTS", "lits"), ("KiTS23", "kits"), ("MSD Pancreas", "msd"),
            ("FLARE-Task2", "flare_task2"), ("FLARE23", "flare23")]
TUMOR_DS = [("LiTS", "lits"), ("MSD Pancreas", "pancreas"), ("KiTS23", "kits"), ("FLARE23", "flare")]


def split_one(cases):
    """filter-then-split for one dataset: fresh RandomState(42), 20% test / 10% val by whole patient."""
    rng = np.random.RandomState(42)
    cs = sorted(cases); rng.shuffle(cs)
    n = len(cs); nte = max(1, int(0.2 * n)); nva = max(1, int(0.1 * n))
    return set(cs[nte + nva:]), set(cs[nte:nte + nva]), set(cs[:nte])   # tr, va, te


def table(meta, datasets):
    slices = defaultdict(lambda: defaultdict(int))
    for m in meta:
        slices[m["dataset"]][m["case"]] += 1
    L = ["| Dataset | Total patients | Train (pt / slices) | Val (pt / slices) | Test (pt / slices) |",
         "|---|:--:|:--:|:--:|:--:|"]
    for name, ds in datasets:
        by = slices[ds]
        tr, va, te = split_one(by.keys())
        s = lambda S: sum(by[c] for c in S)
        L.append(f"| **{name}** | {len(by)} | {len(tr)} / {s(tr):,} | {len(va)} / {s(va):,} | {len(te)} / {s(te):,} |")
    return "\n".join(L)


def build():
    org = json.load(open(ORGAN_META))
    tum = []
    for p in TUMOR_METAS:
        tum += json.load(open(p))
    return ("**Organ models** (`organ_pool_lkp`):\n\n" + table(org, ORGAN_DS)
            + "\n\n**Tumor models** (`tumor_pool` + `flare_tumor_pool` + `kits_tumor_pool`):\n\n"
            + table(tum, TUMOR_DS)
            + "\n\nSplit **by whole patient** (seed 42), 20% test / 10% val; val and test are held-out "
              "patients (no slice-level leakage).")


def main():
    body = build()
    print(body)
    txt = open(README).read()
    if START in txt and END in txt:
        i, j = txt.index(START) + len(START), txt.index(END)
        open(README, "w").write(txt[:i] + "\n" + body + "\n" + txt[j:])
        print("\n-> README splits section updated.")
    else:
        print("\n(markers not found in README — printed only)")


if __name__ == "__main__":
    main()
