#!/usr/bin/env python3
"""Fill the patient-level 3-D and knowledge-graph result sections of README.md from results/*.json.
Marker regions: <!-- RESULTS3D:START/END --> and <!-- KG:START/END -->. Re-run as results land. CPU-only.
"""
import glob
import json
import os
import statistics

README = "/home/ud3d4/Desktop/SWOG/README.md"
RES = "/home/ud3d4/Desktop/SWOG/results"
DS_NAME = {"flare_task2": "FLARE-Task2", "flare23": "FLARE23", "kits": "KiTS23", "msd": "MSD", "lits": "LiTS"}
ORDER = ["flare_task2", "flare23", "kits", "msd", "lits"]


def replace(txt, start, end, body):
    i, j = txt.index(start) + len(start), txt.index(end)
    return txt[:i] + "\n" + body + "\n" + txt[j:]


def results3d():
    L = ["**Patient-level 3-D (whole-volume, semi-oracle setting)** — DSC = volume overlap, NSD@2mm = surface "
         "agreement, HD95 = 95th-percentile boundary distance (mm):", "",
         "| Dataset | Organ | DSC | NSD@2mm | HD95 (mm) | n |", "|---|---|:--:|:--:|:--:|:--:|"]
    for ds in ORDER:
        f = f"{RES}/ausam_3d_{ds}.json"
        if not os.path.exists(f):
            continue
        d = json.load(open(f)); nsd = d.get("mean_nsd_2mm", {}); hd = d.get("mean_hd95_mm", {})
        for organ, dsc in d["mean_3d_dice"].items():
            L.append(f"| {DS_NAME.get(ds, ds)} | {organ} | {dsc:.3f} | "
                     f"{nsd.get(organ, '—') if not isinstance(nsd.get(organ), float) else f'{nsd[organ]:.3f}'} | "
                     f"{hd.get(organ, '—') if not isinstance(hd.get(organ), float) else f'{hd[organ]:.2f}'} | {d['n_patients']} |")
    L.append("\n_LiTS is DSC-only — its `.npy` volumes did not retain mm spacing, which NSD/HD95 require._")
    return "\n".join(L)


def kg():
    L = []
    sm = f"{RES}/ausam_3d_summary.json"
    if os.path.exists(sm):
        nf = json.load(open(sm)).get("node_fidelity", {})
        L += ["**Node fidelity** — is a KG built from *predicted* masks as trustworthy as one from GT? "
              "Predicted-vs-GT organ-volume agreement:", "",
              "| organ / dataset | volume corr | MAPE |", "|---|:--:|:--:|"]
        for k, v in sorted(nf.items()):
            L.append(f"| {k} | {v['volume_corr']} | {v['volume_MAPE_pct']}% |")
        L.append("")
    rf = f"{RES}/retrieval_on_predicted.json"
    if os.path.exists(rf):
        r = json.load(open(rf))["results"]
        L += ["**OAKG retrieval on predicted phenotypes** — ranked by predicted-phenotype similarity, relevance "
              "scored against GT tumor features:", "",
              "| Setting | P@5 | P@10 | mAP | nDCG | spurious@10 |", "|---|:--:|:--:|:--:|:--:|:--:|"]
        for k, v in r.items():
            L.append(f"| {k.split('_', 1)[1].replace('_', ' ')} | {v['P@5']} | {v['P@10']} | {v['mAP']} | "
                     f"{v['nDCG']} | {v['spurious@10']} |")
    else:
        L.append("_OAKG retrieval on predicted phenotypes — computing (pipeline chain running); table lands on completion._")
    return "\n".join(L)


def crossdataset():
    f = f"{RES}/crossdataset_organ.json"
    if not os.path.exists(f):
        return "_Cross-dataset organ matrix pending._"
    M = json.load(open(f))["matrix"]
    L = ["**Cross-dataset organ generalization** — rows = training dataset, columns = test dataset, slice-level Dice "
         "(GT box). **Bold** = in-distribution (diagonal); off-diagonal = zero-shot transfer.", ""]
    on, off = [], []
    for organ in ["liver", "kidney", "pancreas"]:
        if organ not in M:
            continue
        mat = M[organ]
        tests = sorted({t for row in mat.values() for t in row})
        L += [f"_{organ}_", "| model \\ test | " + " | ".join(DS_NAME.get(t, t) for t in tests) + " |",
              "|---|" + "|".join([":--:"] * len(tests)) + "|"]
        for tr in sorted(mat):
            cells = []
            for t in tests:
                v = mat[tr].get(t)
                if v is None:
                    cells.append("—")
                else:
                    cells.append(f"**{v:.3f}**" if t == tr else f"{v:.3f}")
                    (on if t == tr else off).append(v)
            L.append(f"| {DS_NAME.get(tr, tr)} | " + " | ".join(cells) + " |")
        L.append("")
    L.append(f"Mean in-distribution Dice **{statistics.mean(on):.3f}** vs zero-shot transfer "
             f"**{statistics.mean(off):.3f}** (Δ {statistics.mean(on) - statistics.mean(off):+.3f}). Liver transfers "
             f"cleanly (all ≥0.92); the widest gap is transfer *into* KiTS (FLARE→KiTS kidney 0.80–0.85), reflecting "
             f"KiTS's tumor-distorted kidneys and a harder test split — a data-distribution effect that motivates "
             f"integrating datasets at the KG level rather than expecting one model to cover all.")
    return "\n".join(L)


TUMOR_NAME = {"pancreas": "MSD", "lits": "LiTS", "kits": "KiTS23", "flare": "FLARE23"}


def tumor_incremental():
    f = f"{RES}/tumor_incremental.json"
    if not os.path.exists(f):
        return "_Incremental tumor experiment pending._"
    stages = json.load(open(f))["stages"]
    cols = ["pancreas", "lits", "kits", "flare"]
    L = ["**Incremental cross-dataset tumor Dice** — one SAM3 tumor model trained on a *growing* set of datasets, "
         "tested on all four. Rows = cumulative training set; **bold** = the newly-added dataset (in-distribution); "
         "off-diagonal = held-out cross-dataset transfer.", "",
         "| trained on | " + " | ".join(TUMOR_NAME[c] for c in cols) + " |",
         "|---|" + "|".join([":--:"] * len(cols)) + "|"]
    prev = set()
    for s in stages:
        added = next(iter(set(s["trained_on"]) - prev), None)
        prev = set(s["trained_on"])
        label = f"+ {TUMOR_NAME.get(added, added)}" if len(s["trained_on"]) > 1 else TUMOR_NAME.get(added, added)
        cells = []
        for c in cols:
            v = s["per_dataset_test"].get(c)
            cells.append("—" if v is None else (f"**{v:.3f}**" if c == added else f"{v:.3f}"))
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(L)


def tumor_fidelity():
    import numpy as np
    L = ["**Tumor phenotype fidelity** — predicted vs GT over the tumor-test patients (the tumor half of node "
         "fidelity — the categorical phenotypes here are what the KG retrieves on):", "",
         "| dataset / organ | n | tumor-vol corr | has-tumor acc | burden acc | multiplicity acc |",
         "|---|:--:|:--:|:--:|:--:|:--:|"]
    for ds, organ in [("msd", "pancreas"), ("lits", "liver"), ("kits", "kidney")]:
        pf, gf = f"{RES}/corpus_predicted_{ds}.json", f"{RES}/corpus_gt_{ds}.json"
        if not (os.path.exists(pf) and os.path.exists(gf)):
            continue
        P = {r["case_id"]: r for r in json.load(open(pf))["records"]}
        G = {r["case_id"]: r for r in json.load(open(gf))["records"]}
        ids = [i for i in P if i in G]
        po, go = (lambda i: P[i]["organs"][organ]), (lambda i: G[i]["organs"][organ])
        pv = [po(i).get("tumor_volume_cm3") or 0 for i in ids]
        gv = [go(i).get("tumor_volume_cm3") or 0 for i in ids]
        corr = float(np.corrcoef(pv, gv)[0, 1])
        acc = lambda key: float(np.mean([po(i).get(key) == go(i).get(key) for i in ids]))
        ht = float(np.mean([bool(po(i).get("has_tumor")) == bool(go(i).get("has_tumor")) for i in ids]))
        L.append(f"| {DS_NAME.get(ds, ds)} / {organ} | {len(ids)} | {corr:.3f} | {ht:.2f} | "
                 f"{acc('burden_cat'):.2f} | {acc('multiplicity'):.2f} |")
    return "\n".join(L)


def tumor3d():
    d2map = {"msd": "pancreas", "lits": "lits", "kits": "kits", "flare23": "flare"}
    organ = {"msd": "pancreas", "lits": "liver", "kits": "kidney", "flare23": "pan-cancer"}
    fmt = lambda v, p: (f"{v:.{p}f}" if isinstance(v, (int, float)) else "—")
    L = ["**Patient-level 3-D tumor (whole-volume, semi-oracle setting)** — the 3-D counterpart to the §4 2-D "
         "tumor Dice, in the same GT-box / target-present-slice setting as the organ 3-D. DSC / NSD@2mm / HD95, "
         "with the 2-D Dice alongside for reference:", "",
         "| Dataset | Tumor | DSC 3-D | NSD@2mm | HD95 (mm) | 2-D Dice | n |",
         "|---|---|:--:|:--:|:--:|:--:|:--:|"]
    for ds in ["msd", "lits", "kits", "flare23"]:
        f = f"{RES}/tumor_3d_{ds}.json"
        if not os.path.exists(f):
            continue
        t = json.load(open(f))
        d2f = f"{RES}/tumor_ausam_{d2map[ds]}.json"
        two = json.load(open(d2f)).get("tumor_ausam_dice") if os.path.exists(d2f) else None
        n = t["n_patients"]; nstr = f"{n}\\*" if ds == "flare23" else str(n)
        L.append(f"| {DS_NAME.get(ds, ds)} | {organ[ds]} | {fmt(t['mean_3d_dice'], 3)} | "
                 f"{fmt(t['mean_nsd_2mm'], 3)} | {fmt(t['mean_hd95_mm'], 2)} | {fmt(two, 3)} | {nstr} |")
    L += ["", "_LiTS is DSC-only (no mm spacing). **\\*FLARE23 3-D is on only the 8 tumor-test patients with local "
          "volumes** (of 54), so it reads high on an easy subset — its full-set **2-D 0.803** stays the "
          "representative FLARE23 tumor number. Moving tumor from 2-D-target-present to 3-D-whole-volume trims DSC "
          "by ~1.5–6.4 points (MSD −6.4, LiTS −3.4, KiTS −1.5) — modest because, like the organ 3-D, the "
          "semi-oracle setting scores only tumor-present slices with a GT box; the larger drop expected under "
          "autonomous localization (auto-box) is the next experiment._"]
    return "\n".join(L)


def main():
    txt = open(README).read()
    txt = replace(txt, "<!-- RESULTS3D:START -->", "<!-- RESULTS3D:END -->", results3d())
    txt = replace(txt, "<!-- CROSSDATASET:START -->", "<!-- CROSSDATASET:END -->", crossdataset())
    txt = replace(txt, "<!-- TUMOR3D:START -->", "<!-- TUMOR3D:END -->", tumor3d())
    txt = replace(txt, "<!-- TUMOR_INCR:START -->", "<!-- TUMOR_INCR:END -->", tumor_incremental())
    txt = replace(txt, "<!-- TUMOR_FID:START -->", "<!-- TUMOR_FID:END -->", tumor_fidelity())
    txt = replace(txt, "<!-- KG:START -->", "<!-- KG:END -->", kg())
    open(README, "w").write(txt)
    print("README 3-D + KG results updated.")


if __name__ == "__main__":
    main()
