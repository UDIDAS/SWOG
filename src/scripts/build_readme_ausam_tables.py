#!/usr/bin/env python3
"""Fill the AUSAM baseline tables in README.md from the per-dataset AUSAM result JSONs.
Reads results/organ_generic_ausam_<ds>.json (per_organ_dataset) and results/tumor_ausam_<ds>.json
(tumor_ausam_dice), writes markdown between the <!-- AUSAM_BASELINE:START/END --> markers. Re-run as
models finish; cells not yet trained show '…', structurally-absent (organ,dataset) combos show '—'. CPU-only.
"""
import glob
import json
import os

README = "/home/ud3d4/Desktop/SWOG/README.md"
RES = "/home/ud3d4/Desktop/SWOG/results"
START, END = "<!-- AUSAM_BASELINE:START -->", "<!-- AUSAM_BASELINE:END -->"

ORG_DS = [("FLARE23", "flare23"), ("FLARE-Task2", "flare_task2"), ("LiTS", "lits"), ("KiTS", "kits"), ("MSD", "msd")]
ORGANS = ["liver", "kidney", "pancreas"]
EXPECT = {"liver": {"flare23", "flare_task2", "lits"},
          "kidney": {"flare23", "flare_task2", "kits"},
          "pancreas": {"flare23", "flare_task2", "msd"}}
TUM_DS = [("LiTS", "lits"), ("MSD Pancreas", "pancreas"), ("KiTS", "kits"), ("FLARE23", "flare")]


def mean(vals):
    vals = [v for v in vals if v is not None]
    return f"**{sum(vals)/len(vals):.3f}**" if vals else "…"


def build():
    # organ cells
    cell = {}
    for f in glob.glob(f"{RES}/organ_generic_ausam_*.json"):
        for k, v in json.load(open(f)).get("per_organ_dataset", {}).items():
            organ, ds = k.split("/")
            cell[(organ, ds)] = v
    # tumor cells
    tcell = {}
    for _, ds in TUM_DS:
        p = f"{RES}/tumor_ausam_{ds}.json"
        if os.path.exists(p):
            tcell[ds] = json.load(open(p)).get("tumor_ausam_dice")

    L = ["**Organ AUSAM** — held-out per-dataset test Dice (GT-box, per dataset):", ""]
    L.append("| Organ | " + " | ".join(n for n, _ in ORG_DS) + " | **mean** |")
    L.append("|---|" + ":--:|" * (len(ORG_DS) + 1))
    for o in ORGANS:
        row, present = [], []
        for _, ds in ORG_DS:
            if (o, ds) in cell:
                row.append(f"{cell[(o, ds)]:.3f}"); present.append(cell[(o, ds)])
            else:
                row.append("…" if ds in EXPECT[o] else "—")
        L.append(f"| **{o.capitalize()}** | " + " | ".join(row) + f" | {mean(present)} |")

    L += ["", "**Tumor AUSAM** — held-out per-dataset test Dice (GT-box, per dataset):", ""]
    L.append("| " + " | ".join(n for n, _ in TUM_DS) + " | **mean** |")
    L.append("|" + ":--:|" * (len(TUM_DS) + 1))
    trow, tp = [], []
    for _, ds in TUM_DS:
        if ds in tcell and tcell[ds] is not None:
            trow.append(f"{tcell[ds]:.3f}"); tp.append(tcell[ds])
        else:
            trow.append("…")
    L.append("| " + " | ".join(trow) + f" | {mean(tp)} |")

    n_org = len({k[1] for k in cell})
    n_tum = len(tcell)
    L += ["", f"_{len(cell)} organ (dataset,organ) cells from {n_org}/5 datasets · {n_tum}/4 tumor datasets. "
          "'…' = still training, '—' = organ not in that dataset._"]
    return "\n".join(L)


def main():
    txt = open(README).read()
    i, j = txt.index(START) + len(START), txt.index(END)
    txt = txt[:i] + "\n" + build() + "\n" + txt[j:]
    open(README, "w").write(txt)
    print("README AUSAM tables updated.")


if __name__ == "__main__":
    main()
