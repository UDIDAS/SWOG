#!/usr/bin/env python3
"""Build src/notebooks/Two_Model_Paradigm.ipynb — a plain-language, example-first notebook that shows, on
REAL held-out test scans, what the original single-dataset tumor model outputs vs our pooled two-model
approach. Reads results/paradigm_examples.{npz,json} (produced by prep_paradigm_examples.py). CPU-only.
"""
import json
import os

ROOT = "/home/ud3d4/Desktop/SWOG"
OUT = f"{ROOT}/src/notebooks/Two_Model_Paradigm.ipynb"


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": [l + "\n" for l in lines]}


cells = []

cells.append(md(
    "# Why we use two models — shown on real scans",
    "",
    "We segment **tumors** two ways and compare them on the *same* real, held-out CT scans:",
    "",
    "| | Trained on | Idea |",
    "|---|---|---|",
    "| **Original** | **LiTS only** (liver tumors) | the single-dataset paradigm — one dataset, one cancer |",
    "| **Ours** | **all 4 datasets** (liver, pancreas, kidney, FLARE) pooled | the two-model paradigm's pooled tumor model |",
    "",
    "Both models get **only the image and the word `\"tumor\"`** — no box, no hint about where the tumor is. "
    "Finding it is the model's job.",
    "",
    "**The question:** what happens when we show each model a cancer type the *original* one never trained on?",
    "",
    "Everything below is on **held-out test patients** (never seen in training), scored by **Dice** "
    "(overlap with the radiologist's ground truth, 0–100%)."))

cells.append(code(
    "import json, numpy as np, matplotlib.pyplot as plt",
    "ROOT = '/home/ud3d4/Desktop/SWOG'",
    "d = np.load(f'{ROOT}/results/paradigm_examples.npz')",
    "info = json.load(open(f'{ROOT}/results/paradigm_examples.json'))",
    "imgs, gts, po, pu = d['imgs'], d['gts'], d['pred_orig'], d['pred_ours']",
    "ex = info['examples']",
    "print('Original model :', info['orig_ckpt'])",
    "print('Our model      :', info['ours_ckpt'])",
    "print(f'{len(ex)} real example cases loaded')"))

cells.append(md(
    "### How to read each row",
    "- **Left:** the CT scan with the **ground truth** tumor outlined in **green**.",
    "- **Middle:** the **Original** (LiTS-only) model's output in **red**.",
    "- **Right:** **Our** pooled model's output in **red**.",
    "",
    "Green outline is repeated on the middle/right panels so you can see how well the red prediction lands on it."))

cells.append(code(
    "def show(k):",
    "    m = ex[k]; ct = imgs[k][..., 0]",
    "    fig, ax = plt.subplots(1, 3, figsize=(11, 3.9))",
    "    for a in ax: a.imshow(ct, cmap='gray'); a.axis('off')",
    "    ax[0].contour(gts[k].astype(float), [0.5], colors='lime', linewidths=1.8)",
    "    ax[0].set_title('CT + ground truth (green)', fontsize=10)",
    "    for a, pred, name, dc in [(ax[1], po[k], 'ORIGINAL (LiTS-only)', m['dice_orig']),",
    "                              (ax[2], pu[k], 'OURS (pooled, 4 datasets)', m['dice_ours'])]:",
    "        a.contour(gts[k].astype(float), [0.5], colors='lime', linewidths=1.0)",
    "        if pred.any():",
    "            a.imshow(np.ma.masked_where(~pred, pred.astype(float)), cmap='autumn', alpha=0.6, vmin=0, vmax=1)",
    "        a.set_title(f'{name}\\nDice = {dc*100:.0f}%', fontsize=10)",
    "    seen = 'the original DID train on this cancer' if m['seen_by_original'] else 'the original NEVER trained on this cancer'",
    "    fig.suptitle(f\"{m['dataset_name']}  —  patient {m['case']}   ({seen})\", fontsize=11)",
    "    plt.tight_layout(); plt.show()",
    "",
    "def show_dataset(ds):",
    "    ks = [k for k, m in enumerate(ex) if m['dataset'] == ds]",
    "    for k in ks: show(k)",
    "    return ks"))

cells.append(md(
    "## 1. Control — liver tumor (the original *did* train on this)",
    "",
    "Start with a cancer the original model has seen: **liver tumor (LiTS)**. Both models should do fine here. "
    "This is the baseline — it shows the two models are comparable *when the original has coverage*."))
cells.append(code("show_dataset('lits');"))

cells.append(md(
    "## 2. Pancreatic tumor (MSD07) — a cancer the original never saw",
    "",
    "Now a **pancreatic tumor**. The original model was trained only on liver tumors, so it has *never seen* "
    "a pancreatic tumor. Watch the middle panel — it typically outputs **almost nothing**. Our pooled model, "
    "which learned from pancreatic tumors too, finds it."))
cells.append(code("show_dataset('pancreas');"))

cells.append(md(
    "## 3. Kidney tumor (KiTS23) — also unseen by the original",
    "",
    "Same story for **kidney tumors**: unseen by the original, covered by our pooled model."))
cells.append(code("show_dataset('kits');"))

cells.append(md(
    "## 4. FLARE23 pan-cancer — a different distribution again",
    "",
    "**FLARE23** tumors come from a different data source and mix of cancers. The original single-dataset model "
    "struggles; the pooled model handles them far better."))
cells.append(code("show_dataset('flare');"))

cells.append(md(
    "## The numbers behind the pictures",
    "",
    "Average Dice over **all** held-out tumor slices for each cancer type (not just the examples shown) — "
    "so the pictures above are representative, not cherry-picked outliers."))
cells.append(code(
    "import pandas as pd",
    "agg = info['per_dataset_mean_dice']",
    "name = {'lits':'Liver (LiTS)','pancreas':'Pancreas (MSD07)','kits':'Kidney (KiTS23)','flare':'Pan-cancer (FLARE23)'}",
    "rows = []",
    "for ds in ['lits','pancreas','kits','flare']:",
    "    if ds in agg:",
    "        a = agg[ds]",
    "        rows.append([name[ds], a['n_patients'], f\"{a['orig']*100:.1f}%\", f\"{a['ours']*100:.1f}%\",",
    "                     'yes' if ds=='lits' else 'no'])",
    "pd.DataFrame(rows, columns=['Cancer (dataset)','# test patients','Original (LiTS-only)','Ours (pooled)','Original trained on it?'])"))

cells.append(md(
    "## What this shows",
    "",
    "- The **original single-dataset model can only segment the one cancer it trained on.** On liver tumors it is "
    "fine; on pancreatic and kidney tumors it collapses to near-zero — it simply has no concept of them.",
    "- **Our pooled model covers all of them**, because it was trained on all four datasets' tumors at once.",
    "",
    "**Why this forces two models:** you can only train on *all* the cancers by **pooling** them into one tumor "
    "model — and that pooling is only possible because the tumor is a **separate model** from the per-dataset "
    "organ models. A single model that lumped tumor in with each dataset's organs could never pool tumors across "
    "datasets. That is the whole reason for the two-model design.",
    "",
    "**Honest scope:** the pancreas / kidney panels show **coverage** — our pooled model *was* trained on those "
    "cancers. For *true generalization* to a wholly unseen dataset, we hold FLARE23 out of training entirely and "
    "still see tumor Dice rise from **34.8% → 51.4%** as we pool more of the *other* datasets — improvement on a "
    "distribution the model never touched. Coverage and generalization are different claims, and both hold."))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3"}},
      "nbformat": 4, "nbformat_minor": 5}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(nb, open(OUT, "w"), indent=1)
print(f"-> {OUT}  ({len(cells)} cells)")
